import sys
sys.path.insert(0, "../../")
from hpo.promptTaskMix.promptopt.instantiate import GluePromptOpt
from hpo.promptTaskMix.promptopt.techniques.common_logic import DatasetSpecificProcessing
from hpo.promptTaskMix.common.utils.file import save_jsonlist
import os
import re
from tqdm import tqdm
from typing import Any, Dict, List, Iterable
import json
import gzip
from openai import OpenAI
import jsonlines
import yaml
from dotenv import load_dotenv
load_dotenv(override = True)

use_openai = os.getenv("USE_OPENAI_API_KEY", "False").lower() == "true"
use_deepseek = os.getenv("USE_DEEPSEEK_API_KEY", "False").lower() == "true"

if use_openai:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME")
elif use_deepseek:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model_name = os.getenv("DEEPSEEK_MODEL_NAME")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
else:
    raise ValueError("Neither OpenAI nor Deepseek API selected.")

task_details = {
    'informal_to_formal': {
        'task_description': 'In this task, you will be given a sentence in an informal style. Your job is to rewrite the sentence in a formal style.',
        'base_instruction': 'For each given sentence, provide a formal paraphrase.',
        'answer_format': 'For each input sentence, present the reasoning followed by the format paraphrased sentence.'
    },
    'letters_list': {
        'task_description': 'In this task, you will be given a single word as input. Your job is to produce the output by adding a space between each character pair in the word.',
        'base_instruction': 'For each given word, insert a space between each character pair in the word.',
        'answer_format': 'For each input word, output only the space-separated characters.'
    },
    'negation': {
        'task_description': 'For each input, write a sentence that expresses the exact opposite meaning of the input.',
        'base_instruction': 'For each given sentence, provide a new sentence that conveys the exact opposite meaning by using "not" in the input sentence, keeping the rest of the sentence unchanged.',
        'answer_format': "For each input sentence, negate the meaning by adding 'not' to the input sentence."
    },
    'orthography_starts_with': {
        'task_description': 'For each input, output all the words in the sentence that begin with the character in brackets at the end of the sentence.',
        'base_instruction': 'Output words with space-separated that begin with the character in brackets at the end of the following sentence=',
        'answer_format': 'For each input sentence, present the reasoning followed by space-separated words.'
    },
    'rhymes': {
        'task_description': 'In this task, you will be given a single word as input. Your job is to produce a list of comma-separated words that rhyme with the input word.',
        'base_instruction': 'For each given word, provide a list of words that rhyme with the input word=',
        'answer_format': 'For each input word, present the reasoning followed by the list of rhyming words.'
    },
    'second_word_letter': {
        'task_description': 'Extract the second letter from the input word.',
        'base_instruction': 'Output the second letter. Think step by step to arrive at the solution.',
        'answer_format': 'For each input word, present the reasoning followed by the extracted letter (only single letter).'
    },
    'sum': {
        'task_description': 'For each input, write the sum of the two numbers that appear there.',
        'base_instruction': 'Output the sum of the following two numbers=',
        'answer_format': 'For each pair of numbers, present the reasoning followed by the sum.'
    },
    'diff': {
        'task_description': 'For each input, subtract the second number from the first.',
        'base_instruction': 'Output the result of subtracting the second number from the first number=',
        'answer_format': 'For each pair of numbers, present the reasoning followed by the result.'
    },
    'sentence_similarity': {
        'task_description': 'Each input consists of two sentences (Sentence 1 and Sentence 2). Rate on a scale of 0 to 5 whether those sentences are paraphrases of each other, and also give a brief textual description of the rating (0 - definitely not, 2 - possibly, 3 - probably, 4 - almost perfectly and 5 - perfectly). Use "-" to separate them.',
        'base_instruction': '''Rate the similarity of each pair of sentences according to the following scale:

        0 - Definitely not : The sentences are completely unrelated in meaning.
        1 - Probably not : The sentences have minor or superficial similarities but differ significantly in meaning.
        2 - Possibly : The sentences share some elements of meaning but are not strong paraphrases.
        3 - Probably : The sentences convey similar meanings but have some differences.
        4 - Almost perfectly : The sentences are very similar with only minor differences.
        5 - Perfectly : The sentences are nearly identical in meaning.''',
        'answer_format': 'Provide your rating and brief textual description for each pair of sentences from the 6 options. (0 - Definitely not, 1 - Probably not, 2 - Possibly, 3 - Probably, 4 - Almost perfectly, 5 - Perfectly)'
    },
    'test': {
        'task_description': 'Each input consists of two sentences (Sentence 1 and Sentence 2). Rate on a scale of 0 to 5 whether those sentences are paraphrases of each other, and also give a brief textual description of the rating (0 - definitely not, 2 - possibly, 3 - probably, 4 - almost perfectly and 5 - perfectly). Use "-" to separate them.',
        'base_instruction': '''Rate the similarity of each pair of sentences according to the following scale:

        0 - Definitely not : The sentences are completely unrelated in meaning.
        1 - Probably not : The sentences have minor or superficial similarities but differ significantly in meaning.
        2 - Possibly : The sentences share some elements of meaning but are not strong paraphrases.
        3 - Probably : The sentences convey similar meanings but have some differences.
        4 - Almost perfectly : The sentences are very similar with only minor differences.
        5 - Perfectly : The sentences are nearly identical in meaning.''',
        'answer_format': 'Provide your rating and brief textual description for each pair of sentences from the 6 options. (0 - Definitely not, 1 - Probably not, 2 - Possibly, 3 - Probably, 4 - Almost perfectly, 5 - Perfectly)'
    },
    'taxonomy_animal': {
        'task_description': 'In this task, you will be given a list of words. Your job is to identify and list all the animals from the given set of words.',
        'base_instruction': 'For each given list of words, provide a new list containing only the animals.',
        'answer_format': 'For each list of words, output the list of animals.'
    },
    'auto_categorization': {
        'task_description': 'Find the best categorization for the given set of words as input.',
        'base_instruction': 'Output the best categorization for the following set of words=',
        'answer_format': 'For each set of words, present the reasoning followed by the best categorization.'
    },
    'object_counting': {
        'task_description': 'Find the number of objects in the given input.',
        'base_instruction': 'Output the number of objects in the following input=',
        'answer_format': 'For each input, present the reasoning followed by the number of objects.'
    },
    'odd_one_out': {
        'task_description': 'Given the below list of words, find the odd one out.',
        'base_instruction': 'Output the word that does not belong to the group of words=',
        'answer_format': 'For each group of words, present the reasoning followed by the odd one out.'
    },
    'antonyms': {
        'task_description': 'In this task, you will be given a single word as input. Your job is to produce a word that has the exact opposite meaning (an antonym) to the input word.',
        'base_instruction': 'For each given word, provide a word that is an antonym (has the exact opposite meaning).',
        'answer_format': 'For each input word, output only a single word.'
    },
    'word_unscrambling': {
        'task_description': 'In this task output all possible meaningful words that can be formed by rearranging all the letters of the given word. Each character must be used exactly once and the words must be valid.',
        'base_instruction': 'Output all possible meaningful words, comma-separated, that can be formed by rearranging the letters of the given word. Each word must use all the characters from the given word exactly once and must be valid.',
        'answer_format': 'Output all possible meaningful words, comma-separated, that can be formed by rearranging the letters of the given word.'
    },
    'cause_and_effect': {
        'task_description': 'Find the cause in the following cause and effect pair. Each input consists of two sentences, where one is the cause and the other is the outcome.',
        'base_instruction': 'Output the cause in the following cause and effect pair=',
        'answer_format': 'For each pair of sentences, present the reasoning followed by the cause.'
    },
    'common_concept': {
        'task_description': 'In this task, you will be given a list of objects. Your job is to identify and describe a common characteristic that links all the objects in the list.',
        'base_instruction': 'The instruction is to “involve” the objects mentioned in the input.',
        'answer_format': 'For each list of objects, output the common concept by "involving" the objects mentioned.'
    },
    'word_sorting': {
        'task_description': 'In this task, you will be given a set of words. Your job is to sort the words based on the first character of each word in alphabetical order.',
        'base_instruction': 'For each given set of words, provide a sorted list of the words based on the first character of each word.',
        'answer_format': 'For each input, list the sorted words based on the first character of each word.'
    },
    'synonyms': {
        'task_description': 'You will be given a word as input and need to output a word that is semantically similar.',
        'base_instruction': 'Output a word that is semantically similar to the input word=',
        'answer_format': 'For each input word, present the reasoning followed by the synonym.'
    },
    'auto_debugging': {
        'task_description': 'For each input, debug the given program and identify the issue or error present in it. If there is no error, output the result of the last line of the program.',
        'base_instruction': 'Output the error or issue found in the following program, If there is no error, output the result of the last line of the program.',
        'answer_format': 'For each program, present the reasoning followed by the error message or issue.'
    }
}

def update_yaml_file(file_path,config_dict):

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)


    for field,value in config_dict.items():
        data[field] = value

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("YAML file updated successfully!")

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


def call_api(messages):
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
        reasoning_content = ""  
        answer_content = ""  
        is_answering = False 

        for chunk in response:
            delta = chunk.choices[0].delta
            if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
                continue
            if not getattr(delta, 'reasoning_content', None) and not is_answering:
                print("\n" + "=" * 30 + "Response......" + "=" * 30 + "\n")
                is_answering = True
            if getattr(delta, 'reasoning_content', None):
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            elif getattr(delta, 'content', None):
                print(delta.content, end='', flush=True)
                answer_content += delta.content
        return answer_content
    except Exception as e:
        print(f" error : {e}")
        return ""

def llm_eval(predicted_answer, gt_answer):
    EVAL_PROMPT = f"""Given the Predicted_Answer and Reference_Answer, compare them and check they mean the same.
                    If they mean the same then return True between <ANS_START> and <ANS_END> tags , 
                    If they differ in the meaning then return False between <ANS_START> and <ANS_END> tags 
                    Following are the given :
                    Predicted_Answer: {predicted_answer}
                    Reference_Answer: {gt_answer}"""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": EVAL_PROMPT}
    ]

    response = call_api(messages)
    final_judgement = extract_between(start="<ANS_START>", end="<ANS_END>", text=response)
    return final_judgement == "True"


class InstructionInduction(DatasetSpecificProcessing):

    llm_as_judge_eval = None
    taxonomy_animal =False
    orthography_starts_with=False
    sentence_similarity=False
    word_unscrambling=False
    rhymes=False
    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):
            return completion

        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Evaluating samples"):
            example = {
                DatasetSpecificProcessing.QUESTION_LITERAL: sample['question'],
                DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample['answer'],
                DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: extract_answer_from_output(sample["answer"])
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):

        final_answer = extract_between(text=answer, start="<ANS_START>", end="<ANS_END>")
        return final_answer

    def access_answer(self, llm_output: str, gt_answer: str):

        if self.llm_as_judge_eval:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip()
            is_correct = False
            if llm_eval(predicted_answer, gt_answer):
                is_correct = True
        elif self.taxonomy_animal:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip('\"').strip()
            pred_set = set(map(lambda x: x.strip().lower(), predicted_answer.split(',')))
            gt_set = set(map(lambda x: x.strip().lower(), gt_answer.split(',')))
            is_correct = False
            if pred_set == gt_set:
                is_correct = True
        elif self.orthography_starts_with:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip('\"').strip()
            pred_set = set(map(lambda x: x.strip().lower(), predicted_answer.split()))
            gt_set = set(map(lambda x: x.strip().lower(), gt_answer.split()))
            is_correct = False
            if pred_set == gt_set:
                is_correct = True
        elif self.sentence_similarity:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip('\"').strip()
            pattern = r'\b([0-5])\s*-\s*(Definitely not|Probably not|Possibly|Probably|Almost perfectly|Perfectly)\b'
            match = re.search(pattern, predicted_answer, re.IGNORECASE)
            extracted = ""
            is_correct = False
            if match:
                extracted = f"{match.group(1)} - {match.group(2)}"
                if predicted_answer and (extracted.lower() == gt_answer.lower()):
                    is_correct = True
            elif predicted_answer and (predicted_answer == gt_answer.split('-')[0].strip()):
                is_correct = True
        elif self.word_unscrambling:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip('\"').strip()
            is_correct = False
            predicted_parts = [part.strip().lower() for part in predicted_answer.split(",")]
            if gt_answer.lower() in predicted_parts:
                is_correct = True
        elif self.rhymes:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip('\"').strip()
            pred_set = set(map(lambda x: x.strip().lower(), predicted_answer.split(",")))
            gt_set = set(map(lambda x: x.strip().lower(), gt_answer.split(",")))
            is_correct = False
            if gt_set.issubset(pred_set):
                is_correct = True
        else:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip('\"').strip()
            is_correct = False
            if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
                is_correct = True

        return is_correct, predicted_answer


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


if __name__ == '__main__':

    def get_best_prompt(sample, prompt_level, gp: GluePromptOpt):

        best_prompt = gp.get_best_prompt(sample, prompt_level=prompt_level)

        return best_prompt

    def write_results(file_name, sample, idx, best_prompt, reasoning_content, answer_content, predicted_answer, check_result, prompt_level, completion_tokens,prompt_tokens,total_tokens, gp):
        result = {
            'id': idx+1,
            'best_prompt': best_prompt,
            'reasoning_content': reasoning_content,
            'answer_content': answer_content,
            'predicted_answer': predicted_answer,
            'gt_answer': sample['final_answer'],
            'passed': check_result,
            'prompt_level': prompt_level,
            'base_instruction': gp.prompt_opt_param.base_instruction,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens
        }
        with jsonlines.open(file_name+'_with_best_prompt_results.jsonl', mode='a') as writer:
            writer.write(result)

    def generate_with_LLM(model_name, messages, index=None):
        try:
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            stream=False
            )
            answer_content = response.choices[0].message.content

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            print("answer_content：", answer_content)
            print("Prompt Tokens:", prompt_tokens)
            print("Completion Tokens:", completion_tokens)
            print("Total Tokens:", total_tokens)
            answer_content = answer_content.replace("</ANS_END>", "<ANS_END>")
            return answer_content, completion_tokens, prompt_tokens, total_tokens
        except Exception as e:
            print(f"Error processing task {index+1}: {e}")
            error_processing_samples.add(index+1)


    output_format = (
        "Your final answer must be strictly enclosed between <ANS_START> and <ANS_END>. "
        "Do NOT include any labels such as 'Formal:', 'Answer:', or similar in the final answer. "
        "While reasoning and explanations can be included in the model's process, the final answer must only contain the pure content without any extra characters, prefixes, or newlines."
    )
    dataset_list = [
                    'informal_to_formal',
                    'letters_list',
                    'negation',
                    'orthography_starts_with',
                    'rhymes',
                    'sum',
                    'diff',
                    'sentence_similarity',
                    'taxonomy_animal',
                    'auto_categorization',
                     'object_counting',
                    'odd_one_out',
                     'antonyms',
                    'word_unscrambling',
                    'cause_and_effect',
                    'word_sorting',
                    'synonyms',
                    'auto_debugging',
                    'second_word_letter',
                    'common_concept',
                    ]
    for dataset_to_run in dataset_list:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(base_dir)
        data_path = os.path.join(base_dir, "data", dataset_to_run, "test.jsonl")
        data_samples = stream_jsonl(data_path)

        path_to_config = os.path.join(base_dir, "configs")
        promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
        setup_config_path = os.path.join(path_to_config, "setup_config.yaml")
        file_path = os.path.join(base_dir, "configs", "promptopt_config.yaml")

        config_dict = {
            "task_name": dataset_to_run,
            "mutation_rounds": 1,
            "few_shot_count": 3,
            "mutate_refine_iterations": 1,
        }
        update_yaml_file(file_path, config_dict)
        ii_processor = InstructionInduction()
        train_file_name = os.path.join(base_dir, "data", dataset_to_run, "train.jsonl")
        test_file_name = os.path.join(base_dir, "data", dataset_to_run, "test.jsonl")

        gp = GluePromptOpt(promptopt_config_path,
                        setup_config_path,
                        dataset_jsonl=train_file_name,
                        data_processor=ii_processor)

        correct_count = 0
        wrong_count = 0
        failed_samples = set()
        passed_samples = set()
        error_processing_samples = set()
        for index, sample in tqdm(enumerate(list(data_samples)), desc="Processing Samples"):
            if dataset_to_run == 'orthography_starts_with':
                ii_processor.orthography_starts_with = True
            if dataset_to_run == 'taxonomy_animal':
                ii_processor.taxonomy_animal = True
            if dataset_to_run == 'word_unscrambling':
                ii_processor.word_unscrambling = True
            if dataset_to_run == 'rhymes':
                ii_processor.rhymes = True
            # prompt refine
            for prompt_level in range(1,6):
                gp.prompt_opt_param.task_description = task_details[dataset_to_run]['task_description']
                gp.prompt_opt_param.base_instruction = task_details[dataset_to_run]['base_instruction']
                gp.prompt_opt_param.specific_problem = gp.prompt_opt_param.base_instruction + "\nHere's an instance: " + sample['question']
                best_prompt = get_best_prompt(sample, prompt_level, gp)
                final_prompt = best_prompt + '\n' + output_format
                print("\n==============================Final best Prompt==============================")
                print(final_prompt)
                messages = [
                    {"role": "user", "content": final_prompt}
                ]
                answer_content, completion_tokens, prompt_tokens, total_tokens = generate_with_LLM(model_name, messages, index)
                reasoning_content=''
                ii_processor.llm_as_judge_eval = False
                is_correct, predicted_answer = ii_processor.access_answer(answer_content, sample['final_answer'])
                print("==============================Evalutate Results==============================")
                print(is_correct)
                if is_correct:
                    correct_count += 1
                    write_results(dataset_to_run, sample, index, best_prompt, reasoning_content, answer_content, predicted_answer,
                                is_correct, prompt_level, completion_tokens, prompt_tokens, total_tokens, gp)
                    break
                else:
                    if prompt_level < 5:
                        continue
                    else:
                        wrong_count += 1
                        failed_samples.add(index+1)
                        write_results(dataset_to_run, sample, index, best_prompt, reasoning_content, answer_content,
                                    predicted_answer, is_correct, prompt_level, completion_tokens, prompt_tokens, total_tokens, gp)

        print(f"\nTotal Passed: {correct_count}")
        print(f"Total Failed: {wrong_count}")
        print(f"Failed Task IDs: {', '.join(map(str, failed_samples))}")
        print(f"Error Processing Task IDs: {', '.join(map(str, error_processing_samples))}")


