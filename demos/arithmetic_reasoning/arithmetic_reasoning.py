import sys
sys.path.insert(0, "../../")
from hpo.promptTaskMix.promptopt.instantiate import GluePromptOpt
from hpo.promptTaskMix.promptopt.techniques.common_logic import DatasetSpecificProcessing
from hpo.promptTaskMix.common.utils.file import save_jsonlist
from typing import Any, Dict, List, Iterable
from tqdm import tqdm
import json
import yaml
import jsonlines
from openai import OpenAI
import gzip
from re import compile, findall
import os

from dotenv import load_dotenv
load_dotenv(override = True)

task_details = {
    'gsm8k': {
        'task_description': 'You are a mathematics expert.You will be given a mathematics problem which you need to solve.',
        'base_instruction': 'You are a mathematics expert.You will be given a mathematics problem which you need to solve.',
        'answer_format': "Provide a step-by-step reasoning followed by the final answer. The final answer must be strictly enclosed within <ANS_START> and <ANS_END>, and should contain only the exact value—without any additional characters, prefixes, or line breaks."
    },
    'svamp': {
        'task_description': 'You are a mathematics expert.You will be given a mathematics problem which you need to solve.',
        'base_instruction': 'You are a mathematics expert.You will be given a mathematics problem which you need to solve.',
        'answer_format': "Provide a step-by-step reasoning followed by the final answer. The final answer must be strictly enclosed within <ANS_START> and <ANS_END>, and should contain only the exact value—without any additional characters, prefixes, or line breaks."
    },
    'aquarat': {
        'task_description': 'You are a mathematics expert.You will be given a mathematics problem which you need to solve.',
        'base_instruction': 'You are a mathematics expert.You will be given a mathematics problem which you need to solve.',
        'answer_format': "Provide a step-by-step reasoning followed by the final answer. The final answer must be strictly enclosed within <ANS_START> and <ANS_END>, and should consist of only the selected option (i.e., one of A, B, C, D, or E) — without any additional characters, prefixes, or line breaks."
    },
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

class GSM8k(DatasetSpecificProcessing):

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):
            # Your functions for metrics and prompt building
            ans_re = compile(r"#### (\-?[0-9\.\,]+)")
            self.INVALID_ANS = "[invalid]"

            match = ans_re.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return self.INVALID_ANS

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
        
        if not answer:
            return self.INVALID_ANS

        model_pred = answer.lower()
        preds = model_pred.split(self.ANSWER_START.lower())
        answer_flag = True if len(preds) > 1 else False

        pred = preds[-1].replace(",", "")
        pred = [s for s in findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return self.INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]
        return pred
    def access_answer(self, llm_output: str, gt_answer: str):

        predicted_answer = self.extract_final_answer(llm_output)
        is_correct = False
        if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
            is_correct = True

        return is_correct, predicted_answer
class SVAMP(DatasetSpecificProcessing):

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
        
        final_answer = extract_between(text=answer,start="<ANS_START>",end="<ANS_END>")
        return final_answer
    
    def access_answer(self, llm_output: str, gt_answer: str):

        predicted_answer = self.extract_final_answer(llm_output)
        is_correct = False
        if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
            is_correct = True

        return is_correct, predicted_answer

class AQUARAT(DatasetSpecificProcessing):

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
        
        final_answer = extract_between(text=answer,start="<ANS_START>",end="<ANS_END>")
        return final_answer
    
    def access_answer(self, llm_output: str, gt_answer: str):

        predicted_answer = self.extract_final_answer(llm_output)
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
        with jsonlines.open(file_name+'_with_best_prompt_results_DSV3.jsonl', mode='a') as writer:
            writer.write(result)


    def generate_with_LLM(model_name, messages,index=None):
        reasoning_content = ""
        answer_content = ""
        is_answering = False
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"\n{'=' * 30} Call LLM API {'=' * 30}")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                stream_options={
                    "include_usage": True
                }
            )
            print("\n" + "=" * 30 + "Thinking......" + "=" * 30 + "\n")

            for chunk in response:
                if not getattr(chunk, 'choices', None):
                    print("\n" + "=" * 30 + "Token usage" + "=" * 30 + "\n")
                    print(chunk.usage)
                    usage = chunk.usage
                    completion_tokens = usage.completion_tokens
                    prompt_tokens = usage.prompt_tokens
                    total_tokens = usage.total_tokens
                    continue

                delta = chunk.choices[0].delta

                if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
                    continue

                if not getattr(delta, 'reasoning_content', None) and not is_answering:
                    print("\n" + "=" * 30 + "response......" + "=" * 30 + "\n")
                    is_answering = True

                if getattr(delta, 'reasoning_content', None):
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                elif getattr(delta, 'content', None):
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content

            answer_content = answer_content.replace("</ANS_END>", "<ANS_END>")
            return reasoning_content, answer_content, completion_tokens, prompt_tokens, total_tokens
        except Exception as e:
            print(f"Error processing task {index+1}: {e}")
            error_processing_samples.add(index+1)

    dataset_list = [
                'gsm8k',
                'aquarat',
                'svamp'
                ]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(base_dir)
    for dataset_to_run in dataset_list:
        data_path = os.path.join(base_dir, dataset_to_run, "data", "test.jsonl")
        data_samples = list(stream_jsonl(data_path))

        path_to_config = os.path.join(base_dir, dataset_to_run, "configs")
        promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
        setup_config_path = os.path.join(path_to_config, "setup_config.yaml")
        file_path = os.path.join(base_dir, dataset_to_run, "configs", "promptopt_config.yaml")

        config_dict = {
            "task_name": dataset_to_run,
            "task_description": task_details[dataset_to_run]['task_description'],
            "base_instruction": task_details[dataset_to_run]['base_instruction'],
            "mutation_rounds": 1,
            "few_shot_count": 3,
            "mutate_refine_iterations": 1,
        }
        update_yaml_file(file_path, config_dict)
        if dataset_to_run == 'gsm8k':
            data_processor = GSM8k()
        if dataset_to_run == 'aquarat':
            data_processor = AQUARAT()
        if dataset_to_run == 'svamp':
            data_processor = SVAMP()
        train_file_name = os.path.join(base_dir, dataset_to_run, "data", "train.jsonl")
        test_file_name = os.path.join(base_dir, dataset_to_run, "data", "test.jsonl")

        gp = GluePromptOpt(promptopt_config_path,
                        setup_config_path,
                        dataset_jsonl=train_file_name,
                        data_processor=data_processor)

        correct_count = 0
        wrong_count = 0
        failed_samples = set()
        passed_samples = set()
        error_processing_samples = set()
        for index, sample in tqdm(enumerate(data_samples), desc="Processing Samples"):
            # prompt refine
            for prompt_level in range(1,6):
                gp.prompt_opt_param.task_description = task_details[dataset_to_run]['task_description']
                gp.prompt_opt_param.base_instruction = task_details[dataset_to_run]['base_instruction']
                gp.prompt_opt_param.specific_problem = gp.prompt_opt_param.base_instruction + "\nHere's an instance: " + sample['question']
                best_prompt = get_best_prompt(sample, prompt_level, gp)
                output_format = task_details[dataset_to_run]['answer_format']
                final_prompt = best_prompt + '\n' + output_format
                print("\n==============================Final best Prompt==============================")
                print(final_prompt)
                messages = [
                    {"role": "user", "content": final_prompt}
                ]
                reasoning_content, answer_content, completion_tokens, prompt_tokens, total_tokens = generate_with_LLM(model_name, messages, index)
                is_correct, predicted_answer = data_processor.access_answer(answer_content, sample['final_answer'])
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












