import random
import re
from os.path import join
from tqdm import tqdm
from typing import Any, Dict, List, Iterable

from ....paramlogger import ParamLogger
from ....paramlogger.constants import LogLiterals
from ....common.base_classes import SetupConfig, UniversalBaseClass
from ....common.llm.llm_mgr import LLMMgr
from ....common.human_eval import execution
from ....common.constants.log_strings import CommonLogsStr
from ...constants import PromptOptimizationParams, SupportedPromptOpt
from ...techniques.common_logic import DatasetSpecificProcessing, PromptOptimizer
from ...techniques.attr_critique_n_refine.base_classes import AttrCritiqueNRefinePromptPool
from promptwizard.glue.common.human_eval.data import write_jsonl, read_problems


def extract_between(start, end, text):
    """
    Extracts the substring from 'text' that is between 'start' and 'end' strings.

    Parameters:
    - start (str): The starting delimiter string.
    - end (str): The ending delimiter string.
    - text (str): The text to search within.

    Returns:
    - str: The extracted substring between the start and end delimiters.
    """
    start_index = text.find(start)
    if start_index == -1:
        return ''

    start_index += len(start)

    end_index = text.find(end, start_index)
    if end_index == -1:
        return ''
    return text[start_index:end_index]


class AttrCritiqueNRefine(PromptOptimizer, UniversalBaseClass):

    TECHNIQUE_NAME = SupportedPromptOpt.ATTR_CRITIQUE_N_REFINE.value

    class GetPromptScoreIndex:
        """
        Class to hold constants. Output of get_prompt_score() method is a list.
        This class stores mapping between output entity and its index in output of get_prompt_score() method.
        """
        PROMPT_STR = 0
        SCORE = 1
        DATASET = 2

    # This has to defined outside of constructor, so that it can be used as decorator.
    iolog = ParamLogger()

    def __init__(self, dataset: List, base_path: str, setup_config: SetupConfig,
                 prompt_pool: AttrCritiqueNRefinePromptPool, data_processor: DatasetSpecificProcessing, logger):
        self.dataset = dataset
        self.setup_config = setup_config
        self.data_processor = data_processor
        self.logger = logger
        self.prompt_pool = prompt_pool
        base_path = join(base_path, LogLiterals.DIR_NAME)
        self.iolog.reset_eval_glue(base_path)

    @iolog.log_io_params
    def chat_completion(self, user_prompt: str, system_prompt: str = None):
        """
        Make a chat completion request to the OpenAI API.

        :param user_prompt: Text spoken by user in a conversation.
        :param system_prompt: Text spoken by system in a conversation.
        :return: Output of LLM
        """
        if not system_prompt:
            system_prompt = self.prompt_pool.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = LLMMgr.chat_completion(messages)
        return response

    @iolog.log_io_params
    def gen_different_styles(self, task_description: str,
                             mutation_rounds: int = 2, thinking_styles_count: int = 10) -> List:
        candidate_prompts = []

        for mutation_round in range(mutation_rounds):
            mutated_sample_prompt = self.prompt_pool.meta_sample_template.format(
                task_description=task_description,
                thinking_styles="\n".join(self.prompt_pool.thinking_styles[:thinking_styles_count]),
                num_variations=thinking_styles_count,
            )
            generated_mutated_prompt = self.chat_completion(mutated_sample_prompt)
            # Find all matches of the pattern in the text
            matches = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN_MUTATION, generated_mutated_prompt)
            candidate_prompts.extend(matches)

            self.logger.info(f"mutation_round={mutation_round + 1} mutated_sample_prompt={mutated_sample_prompt}"
                             f"mutated_prompt_generation={generated_mutated_prompt}")

        return candidate_prompts

    @iolog.log_io_params
    def critique_and_refine(self, prompt: str, critique_example_set: List, fail_codes=None, fail_reasons=None) -> str:
        if fail_reasons is None:
            fail_reasons = []
        example_string = "\n".join(str(example) for example in critique_example_set)
        fail_codes_string = "\n".join(str(code) for code in fail_codes)
        wrong_reasons_string = "\n".join(str(wrong_reason) for wrong_reason in fail_reasons)
        examples_code_reasons_string = "\n".join(
            f"Example: {example}\nCode: {code}\nReason: {wrong_reason}\n"
            for example, code, wrong_reason in zip(critique_example_set, fail_codes, fail_reasons)
        )

        critique_refine_prompt = self.prompt_pool.critique_refine_template.format(instruction=prompt,
                                                                                  trueorfalse="false",
                                                                                  examples_code_reasons=examples_code_reasons_string,
                                                                                  num_samples=1)

        refined_prompts = self.chat_completion(critique_refine_prompt, self.prompt_pool.expert_profile)
        refined_prompts = refined_prompts.replace("</START>", "<END>")
        refined_prompts = refined_prompts.replace("</END>", "<END>")
        print(
            "***************************************************************************************************************")
        print(refined_prompts)
        print(
            "***************************************************************************************************************")
        refined_prompts = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, refined_prompts)

        if refined_prompts:
            final_refined_prompts = refined_prompts
        else:
            raise ValueError("The LLM ouput is not in the expected format. Please rerun the code...")

        self.logger.info(
            f"Prompt to get Refinement after critique, from LLM:\n {critique_refine_prompt}"
            f"Refined prompts received from LLM:\n {final_refined_prompts}")

        return final_refined_prompts


    @iolog.log_io_params
    def refine_prompts(self, prompt_score_list: List, params: PromptOptimizationParams) -> List:

        refined_prompts = []
        for prompt, score, critique_example_set, *rest in prompt_score_list:
            failed_codes = rest[0] if len(rest) > 0 else None
            failed_reasons = rest[1] if len(rest) > 1 else None
            if score >= params.max_correct_count:
                # if it's good enough prompt
                refined_prompts.append(prompt)
            else:
                # if it's not good enough prompt, how to mutate on that
                new_prompts = self.critique_and_refine(prompt, critique_example_set, failed_codes, failed_reasons)
                if new_prompts:
                    refined_prompts.extend(new_prompts)

        self.logger.info(f"refined_prompts {refined_prompts}")
        return refined_prompts

    @iolog.log_io_params
    def evaluate_MBPP(self, completion: str, sample: Dict) -> Dict:
        """
        Compare predicted answers with actual answers from the dataset.
        Return the list of questions for which the predicted answer was wrong.

        :param generated_text: Output of LLM, that has answers for a mini-batch of questions
                               (which were send in single go)
        :param dataset_subset: List of examples with question and ground truth.
        :return: List of examples that were wrongly classified.
        """
        import threading
        class TimeoutException(Exception):
            pass

        def run_with_timeout(func, args=(), kwargs={}, timeout=2):
            result = {'value': None, 'error': None}

            def target():
                try:
                    result['value'] = func(*args, **kwargs)
                except Exception as e:
                    result['error'] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                raise TimeoutException('Timeout')
            if result['error'] is not None:
                raise result['error']

            return result['value']

        tests = sample['test_list']
        test_imports = sample.get('test_imports', [])
        passed = True
        error_message = ''
        globals_dict = {}
        for import_statement in test_imports:
            try:
                exec(import_statement, globals_dict)
            except Exception as e:
                passed = False
                error_message = f'Error in import: {str(e)}'
                result = {
                    'passed': passed,
                    'error_message': error_message
                }
                return result

        for i, test in enumerate(tests):
            try:
                print(f'Starting test {i + 1}')
                run_with_timeout(exec, (completion, globals_dict), timeout=3)
                run_with_timeout(exec, (test, globals_dict), timeout=3)
                print(f'Finished test {i + 1}')
            except TimeoutException:
                passed = False
                error_message = f'Test case {i + 1} timed out. Test case: {test}'
            except Exception as e:
                passed = False
                error_message = f'Error in test case {i + 1}: {str(e)}. Test case: {test}'

        result = {
            'passed': passed,
            'error_message': error_message
        }

        return result

    @iolog.log_io_params
    def select_top_prompts(self, prompt_score_list: List, top_n: int) -> List:

        sorted_prompts = sorted(
            prompt_score_list,
            key=lambda x: (-x[self.GetPromptScoreIndex.SCORE], len(x[self.GetPromptScoreIndex.PROMPT_STR].split()))
        )
        return sorted_prompts[:top_n]

    def extract_examples_frm_response(self, response_with_examples: str) -> List:

        synthetic_examples = []
        response_with_examples = response_with_examples.replace('</END>', '<END>')
        parsed_data = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, response_with_examples, re.DOTALL)
        parsed_data = [s.strip() for s in parsed_data]

        for text in parsed_data:
            # Splitting text into question, reason, and answer
            if DatasetSpecificProcessing.QUESTION_KEY_IN_PROMPT in text and \
                    DatasetSpecificProcessing.ANSWER_KEY_IN_PROMPT in text:
                question = text[text.find(DatasetSpecificProcessing.QUESTION_KEY_IN_PROMPT) +
                                len(DatasetSpecificProcessing.QUESTION_KEY_IN_PROMPT):
                                text.find(DatasetSpecificProcessing.ANSWER_KEY_IN_PROMPT)].strip()
                answer_with_reason = text[text.find(DatasetSpecificProcessing.ANSWER_KEY_IN_PROMPT) +
                                          len(DatasetSpecificProcessing.ANSWER_KEY_IN_PROMPT):].strip()
                final_answer = extract_between(text=answer_with_reason, start="<ANS_START>", end="<ANS_END>")

                formatted_data = {
                    DatasetSpecificProcessing.QUESTION_LITERAL: question,
                    DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: answer_with_reason,
                    DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: final_answer
                }

                synthetic_examples.append(formatted_data)

        return synthetic_examples


    @iolog.log_io_params
    def generate_expert_identity(self, task_description: str) -> str:
        expert_prompt = self.prompt_pool.expert_template.format(task_description=task_description)
        return self.chat_completion(expert_prompt)

    @iolog.log_io_params
    def generate_knowledge(self, task_description: str):

        prompt_template = self.prompt_pool.gen_knowledge_template.format(specific_problem=task_description)
        return self.chat_completion(user_prompt=prompt_template)

    def generate_best_examples_zero_shot(self, params: PromptOptimizationParams) -> List:

        few_shot_critique_prompt = self.prompt_pool.examples_critique_template_zero_shot. \
            format(task_description=params.task_description,
                   num_examples=params.num_train_examples,
                   specific_problem=params.specific_problem)

        critique = self.chat_completion(few_shot_critique_prompt, self.prompt_pool.expert_profile)
        print("=================================================Critique=============================================")
        print(critique)
        few_shot_opt_prompt = self.prompt_pool.examples_optimization_template. \
            format(gt_example="",
                   critique=critique,
                   task_description=params.task_description,
                   specific_problem=params.specific_problem,
                   num_examples=params.num_train_examples)
        synthetic_examples = self.chat_completion(few_shot_opt_prompt, self.prompt_pool.expert_profile)
        synthetic_examples = self.extract_examples_frm_response(synthetic_examples)
        return synthetic_examples


    @iolog.log_io_params
    def get_samples_scores(self, instructions: List[str], params: PromptOptimizationParams,
                           random_samples=None) -> List:
        results = []
        if random_samples is None:
            mbpp_samples = read_problems(r"../../../../../demos/code_generation/mbpp.jsonl")
            random_samples = random.sample(list(mbpp_samples.values()), params.max_eval_batches)
            random_samples.append(params.curr_sample)
        for instruction in instructions:
            correct_count = 0
            failed_samples = set()
            failed_codes = list()
            failed_samples_reasons = list()
            passed_samples = set()

            for sample in random_samples:
                sample_problem = (sample['text']
                                  + 'Your code should pass these tests:\n\n'
                                  + "\n".join(sample['test_list']) + '\n'
                                  )
                solve_prompt = self.prompt_pool.solve_template.format(
                    questions_batch_size=params.questions_batch_size,
                    instruction=instruction,
                    sample_problem=sample_problem
                )
                print(solve_prompt)
                print("\nGenerating sample code......")

                generated_text = self.chat_completion(solve_prompt)
                generated_text = generated_text.replace("</ANS_END>", "<ANS_END>")
                final_answer = re.findall(DatasetSpecificProcessing.ANSWER_DELIMITER_PATTERN, generated_text)[0]

                evaluate_result = self.evaluate_MBPP(final_answer, sample)
                print("\nEvaluate results......")
                print(evaluate_result)
                if evaluate_result["passed"]:
                    correct_count += 1
                    passed_samples.add(sample['text'])
                else:
                    failed_samples.add(sample['text'])
                    failed_codes.append(final_answer)
                    failed_samples_reasons.append(evaluate_result['error_message'])

            if correct_count == len(random_samples):
                results.append([instruction, correct_count, passed_samples])
            else:
                results.append([instruction, correct_count, failed_samples, failed_codes, failed_samples_reasons])

        return results, random_samples

    def generate_prompt_attribute(self, base_instruction: str):

        prompt_template = self.prompt_pool.gen_attribute_template.format(base_instruction=base_instruction)
        return self.chat_completion(user_prompt=prompt_template)

    def attribute_based_refine(self, base_instruction: str, generated_attributed_prompt: str):

        prompt_template = self.prompt_pool.attribute_based_refine_template.format(base_instruction=base_instruction,
                                                                                  attributed_componment=generated_attributed_prompt)
        return self.chat_completion(user_prompt=prompt_template)

    def get_best_prompt(self, params: PromptOptimizationParams, prompt_level, sample) -> (str, Any):

        params.curr_sample = sample
        params.specific_problem = (sample['text']
                                   + 'Your code should pass these tests:\n\n'
                                   + "\n".join(sample['test_list']) + '\n'
                                   )
        if prompt_level == 1:
            expert_identity = self.generate_expert_identity(params.specific_problem)
            best_prompt = self.prompt_pool.best_prompt.format(expert_identity=expert_identity,
                                                              task_description=params.task_description,
                                                              few_shot_examples='',
                                                              specific_problem=params.specific_problem)
            params.expert_identity = expert_identity
            print("==========================================Generated Expert Identity================================")
            print(params.expert_identity)
            return best_prompt
        if prompt_level == 2:
            print("\nMutating Task Description....")
            # Mutate and refine task description
            for round_num in tqdm(range(1, params.mutate_refine_iterations + 1), desc="Iterations completed: "):
                self.logger.info(f"{CommonLogsStr.LOG_SEPERATOR} + Starting iteration: {round_num} ")
                candidate_prompts = self.gen_different_styles(params.task_description,
                                                              params.mutation_rounds,
                                                              params.style_variation)

                prompt_score_list, random_samples = self.get_samples_scores(candidate_prompts, params)

                if params.refine_instruction:
                    refined_prompts = self.refine_prompts(prompt_score_list, params)
                    refined_prompt_score_list, _ = self.get_samples_scores(refined_prompts, params, random_samples)
                    prompt_score_list = self.select_top_prompts(refined_prompt_score_list,
                                                                params.top_n)

                best_task_description_prompt = prompt_score_list[0][self.GetPromptScoreIndex.PROMPT_STR]
                self.logger.info({"round_num": round_num,
                                  "best_task_description_prompt": best_task_description_prompt,
                                  "score": prompt_score_list[0][self.GetPromptScoreIndex.SCORE]
                                  })
                best_prompt = self.prompt_pool.best_prompt.format(expert_identity=params.expert_identity,
                                                                  task_description=best_task_description_prompt,
                                                                  few_shot_examples='',
                                                                  specific_problem=params.specific_problem)
                params.task_description = best_task_description_prompt
                print(
                    "==========================================Generated Task Description=================================")
                print(params.task_description)
                return best_prompt
        if prompt_level == 3:
            demonstration_examples = []
            print("Generating Demonstration Examples....")
            demonstration_examples = self.generate_best_examples_zero_shot(params)
            formatted_examples = []
            for example in demonstration_examples:
                formatted_example = self.prompt_pool.quest_reason_ans.format(
                    question=example["question"],
                    answer=example["answer"]
                )
                formatted_examples.append(formatted_example)

            formatted_examples_output = "\n\n".join(formatted_examples)
            print("============================Generated Demonstration Examples============================")
            print(formatted_examples_output)

            best_prompt = self.prompt_pool.best_prompt.format(expert_identity=params.expert_identity,
                                                              task_description=params.task_description,
                                                              few_shot_examples=formatted_examples_output,
                                                              specific_problem=params.specific_problem)
            params.demonstration_examples = formatted_examples_output
            return best_prompt

        if prompt_level == 4:
            print("\nGenerating knowledge....")
            generated_knowledge = self.generate_knowledge(params.specific_problem)
            print("============================Generated knowledge============================")
            print(generated_knowledge)
            params.generated_knowledge = generated_knowledge
            best_prompt = self.prompt_pool.best_prompt.format(expert_identity=params.expert_identity,
                                                              task_description=params.task_description,
                                                              few_shot_examples=params.demonstration_examples,
                                                              specific_problem=params.specific_problem)
            knowledge_usage = "Using the following knowledge: "
            best_prompt = best_prompt + '\n' + knowledge_usage + generated_knowledge

            return best_prompt

        if prompt_level == 5:
            print("\nPrompt debugging....")
            generated_attributed_prompt = self.generate_prompt_attribute(sample['text'])
            print("============================Generated Attributed Prompt============================")
            print(generated_attributed_prompt)
            refined_instruction = self.attribute_based_refine(sample['text'], generated_attributed_prompt)
            final_refined_instruction = extract_between(text=refined_instruction, start="<START>", end="END>")
            params.base_instruction = final_refined_instruction
            params.specific_problem = (final_refined_instruction
                                       + 'Your code should pass these tests:\n\n'
                                       + "\n".join(sample['test_list']) + '\n'
                                       )

            best_prompt = self.prompt_pool.best_prompt.format(expert_identity=params.expert_identity,
                                                              task_description=params.task_description,
                                                              few_shot_examples=params.demonstration_examples,
                                                              specific_problem=params.specific_problem)

            knowledge_usage = "Using the following knowledge: "
            best_prompt = best_prompt + '\n' + knowledge_usage + params.generated_knowledge

            return best_prompt
