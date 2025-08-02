import random
import re
from os.path import join
from tqdm import tqdm
from typing import Any, Dict, List, Iterable, Tuple

from ....paramlogger import ParamLogger
from ....paramlogger.constants import LogLiterals
from ....common.base_classes import SetupConfig, UniversalBaseClass
from ....common.llm.llm_mgr import LLMMgr
from ....common.constants.log_strings import CommonLogsStr
from ...constants import PromptOptimizationParams, SupportedPromptOpt
from ...techniques.common_logic import DatasetSpecificProcessing, PromptOptimizer
from ...techniques.attr_sentiment_prompt_refine.base_classes import AttrSentimentRefinePromptPool


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


class AttrSentimentRefine(PromptOptimizer, UniversalBaseClass):
    """
    TODO: Explain this method
    """

    TECHNIQUE_NAME = SupportedPromptOpt.AttrSentimentRefine.value

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
                 prompt_pool: AttrSentimentRefinePromptPool, data_processor: DatasetSpecificProcessing, logger):
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
    def gen_different_styles(self, task_name: str, task_description: str,
                             mutation_rounds: int = 2, thinking_styles_count: int = 10) -> List:
        candidate_prompts = []

        for mutation_round in range(mutation_rounds):
            styles_for_task = self.prompt_pool.thinking_styles.get(task_name, [])
            mutated_sample_prompt = self.prompt_pool.meta_sample_template.format(
                task_description=task_description,
                thinking_styles="\n".join(styles_for_task[:thinking_styles_count]),
                num_variations=thinking_styles_count,
                )
            generated_mutated_prompt = self.chat_completion(mutated_sample_prompt)
            generated_mutated_prompt = generated_mutated_prompt.replace("</END>", "<END>")
            # Find all matches of the pattern in the text
            matches = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN_MUTATION, generated_mutated_prompt)
            candidate_prompts.extend(matches)

            self.logger.info(f"mutation_round={mutation_round+1} mutated_sample_prompt={mutated_sample_prompt}"
                             f"mutated_prompt_generation={generated_mutated_prompt}")

        return candidate_prompts

    @iolog.log_io_params
    def critique_and_refine(self, prompt: str, critique_example_set: List, fail_answers=None, failed_samples_gt_answers=None) -> List:

        if failed_samples_gt_answers is None:
            failed_samples_gt_answers = []

        examples_answer_and_gt_string = "\n".join(
            f"Example: {example}\nPredicted Answer: {answer}\nGround Truth: {gt_answer}\n"
            for example, answer, gt_answer in zip(critique_example_set, fail_answers, failed_samples_gt_answers)
        )

        critique_refine_prompt = self.prompt_pool.critique_refine_template.format(instruction=prompt,
                                                                                  trueorfalse="false",
                                                                                  examples_answer_reasons=examples_answer_and_gt_string,
                                                                                  num_samples=1)

        refined_prompts = self.chat_completion(critique_refine_prompt, self.prompt_pool.expert_profile)
        refined_prompts = refined_prompts.replace("</START>", "<END>")
        refined_prompts = refined_prompts.replace("</END>", "<END>")
        print("***************************************************************************************************************")
        print(refined_prompts)
        print("***************************************************************************************************************")
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
            failed_answers = rest[0] if len(rest) > 0 else None
            failed_samples_gt_answers = rest[1] if len(rest) > 1 else None
            if score >= params.max_correct_count:
                # if it's good enough prompt
                refined_prompts.append(prompt)
            else:
                # if it's not good enough prompt, how to mutate on that
                new_prompts = self.critique_and_refine(prompt, critique_example_set, failed_answers, failed_samples_gt_answers)
                if new_prompts:
                    refined_prompts.extend(new_prompts)

        self.logger.info(f"refined_prompts {refined_prompts}")
        return refined_prompts

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
    def generate_knowledge(self, base_instruction: str):
        prompt_template = self.prompt_pool.gen_knowledge_template.format(base_instruction=base_instruction)
        return self.chat_completion(user_prompt=prompt_template)

    def generate_best_examples_zero_shot(self,params: PromptOptimizationParams) -> List:

        few_shot_critique_prompt = self.prompt_pool.examples_critique_template_zero_shot.\
            format(task_description=params.task_description,
                   num_examples=params.num_train_examples,
                   specific_problem=params.specific_problem)

        critique = self.chat_completion(few_shot_critique_prompt, self.prompt_pool.expert_profile)
        print("=================================================Critique=============================================")
        print(critique)
        few_shot_opt_prompt = self.prompt_pool.examples_optimization_template.\
            format(gt_example="",
                   critique=critique,
                   task_description=params.task_description,
                   specific_problem=params.specific_problem,
                   num_examples=params.num_train_examples)
        synthetic_examples = self.chat_completion(few_shot_opt_prompt, self.prompt_pool.expert_profile)
        synthetic_examples = self.extract_examples_frm_response(synthetic_examples)
        return synthetic_examples

    def get_samples_scores(self, instructions: List[str], params: PromptOptimizationParams, random_samples=None) -> Tuple[List, Any]:
        results = []
        if random_samples is None:
            data_samples = self.dataset
            random_samples = random.sample(data_samples, params.max_eval_batches)
        for instruction in instructions:
            correct_count = 0
            failed_samples = list()  # 存储失败的样本
            failed_answers = list()
            failed_samples_gt_answers = list()
            passed_samples = list()  # 存储通过的样本

            for sample in random_samples:
                sample_problem = params.base_instruction + "\nHere's an instance: " + sample['question']
                solve_prompt = self.prompt_pool.solve_template.format(
                    instruction=instruction,
                    sample_problem=sample_problem
                )
                if params.task_name == "aquarat":
                    solve_prompt += "The final answer must be strictly enclosed within <ANS_START> and <ANS_END>, and should consist of only the selected option (i.e., one of A, B, C, D, or E) — without any additional characters, prefixes, or line breaks."
                if params.task_name == "gsm8k" or params.task_name == "svamp":
                    solve_prompt += "The final answer must be strictly enclosed within <ANS_START> and <ANS_END>, and should contain only the exact value—without any additional characters, prefixes, or line breaks."
                print(solve_prompt)
                print("\nGenerating sample answers......")

                generated_text = self.chat_completion(solve_prompt)
                generated_text = generated_text.replace("</ANS_END>", "<ANS_END>")

                is_correct, predicted_answer = self.data_processor.access_answer(generated_text, sample['final_answer'])
                print("\nEvaluate results......")
                print(is_correct)
                if is_correct:
                    correct_count += 1
                    passed_samples.append(sample['question'])
                else:
                    failed_samples.append(sample['question'])
                    failed_answers.append(predicted_answer)
                    failed_samples_gt_answers.append(sample['final_answer'])

            if correct_count == len(random_samples):
                results.append([instruction, correct_count, passed_samples])
            else:
                results.append([instruction, correct_count, failed_samples, failed_answers, failed_samples_gt_answers])

        return results, random_samples
    
    def generate_prompt_attribute(self, base_instruction: str):

        prompt_template = self.prompt_pool.gen_attribute_template.format(base_instruction=base_instruction)
        return self.chat_completion(user_prompt=prompt_template)
    
    def attribute_based_refine(self, base_instruction: str, generated_attributed_prompt: str):
        
        prompt_template = self.prompt_pool.attribute_based_refine_template.format(base_instruction=base_instruction, attributed_componment=generated_attributed_prompt)
        return self.chat_completion(user_prompt=prompt_template)

    def get_best_prompt(self, params: PromptOptimizationParams, prompt_level, sample) -> (str, Any):

        params.curr_sample = sample
        if prompt_level == 1:
            expert_identity = self.generate_expert_identity(params.task_description)
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
                candidate_prompts = self.gen_different_styles(params.task_name,
                                                              params.task_description,
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
                print("==========================================Generated Task Description=================================")
                print(params.task_description)
                return best_prompt
        if prompt_level == 3:
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
            generated_knowledge = self.generate_knowledge(params.base_instruction)
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
            generated_attributed_prompt = self.generate_prompt_attribute(params.base_instruction)
            print("============================Generated Attributed Prompt============================")
            print(generated_attributed_prompt)
            refined_instruction = self.attribute_based_refine(params.base_instruction, generated_attributed_prompt)
            final_refined_instruction = extract_between(text=refined_instruction, start="<START>", end="END>")
            params.base_instruction = final_refined_instruction
            params.specific_problem = final_refined_instruction + "\nHere's an instance: " + sample['question']
            
            best_prompt = self.prompt_pool.best_prompt.format(expert_identity=params.expert_identity,
                                                    task_description=params.task_description,
                                                    few_shot_examples=params.demonstration_examples,
                                                    specific_problem=params.specific_problem)
            
            knowledge_usage = "Using the following knowledge: "
            best_prompt = best_prompt + '\n' + knowledge_usage + params.generated_knowledge

            return best_prompt

            
