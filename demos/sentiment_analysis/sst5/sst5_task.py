import sys
sys.path.insert(0, "../../../")
from hpo.promptTaskMix.promptopt.instantiate import GluePromptOpt
from hpo.promptTaskMix.promptopt.techniques.common_logic import DatasetSpecificProcessing
from hpo.promptTaskMix.common.utils.file import save_jsonlist
from azure.identity import get_bearer_token_provider, AzureCliCredential
from openai import AzureOpenAI
import os
from tqdm import tqdm
from typing import Any, Dict, List, Iterable
import json
import gzip
from openai import OpenAI
import jsonlines
import yaml
from dotenv import load_dotenv
load_dotenv(override = True)


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
        return text

    start_index += len(start)

    end_index = text.find(end, start_index)
    if end_index == -1:
        return ''
    return text[start_index:end_index]


def call_api(messages):
    token_provider = get_bearer_token_provider(
        AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        api_version="<OPENAI_API_VERSION>",
        azure_endpoint="<AZURE_ENDPOINT>",
        azure_ad_token_provider=token_provider
    )
    response = client.chat.completions.create(
        model="<MODEL_DEPLOYMENT_NAME>",
        messages=messages,
        temperature=0.0,
    )
    prediction = response.choices[0].message.content
    return prediction


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

class SST(DatasetSpecificProcessing):

    llm_as_judge_eval = None

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
            predicted_answer = self.extract_final_answer(llm_output)
            is_correct = False
            if llm_eval(predicted_answer, gt_answer):
                is_correct = True
        else:
            predicted_answer = self.extract_final_answer(llm_output).replace("\n", "").strip()
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

    output_format = "The final answer must be one of the following five: very negative, negative, neutral, positive, or very positive." \
                    "Please make sure that the final answer is wrapped with <ANS_START> and <ANS_END> and does not contain any reasoning or related explanations."

    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(base_dir)
    data_samples = stream_jsonl("data/sst5_test.jsonl")

    path_to_config = "configs"
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")
    file_path = 'configs/promptopt_config.yaml'
    config_dict = {
        "task_description": "Please perform Sentiment Classification task.",
        "mutation_rounds": 1,
        "few_shot_count": 3,
        "mutate_refine_iterations": 1,
    }
    update_yaml_file(file_path, config_dict)
    sst_processor = SST()

    file_path = os.path.join(base_dir, 'data', 'sst5_train.jsonl')

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
                    print("\n" + "=" * 30 + "Response......" + "=" * 30 + "\n")
                    is_answering = True

                if getattr(delta, 'reasoning_content', None):
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                elif getattr(delta, 'content', None):
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content
            answer_content = answer_content.replace("</ANS_END>", "<ANS_END>")
            final_answer = answer_content
            return reasoning_content, answer_content, final_answer, completion_tokens, prompt_tokens, total_tokens
        except Exception as e:
            print(f"Error processing task {index+1}: {e}")
            error_processing_samples.add(index+1)

    gp = GluePromptOpt(promptopt_config_path,
                       setup_config_path,
                       dataset_jsonl=file_path,
                       data_processor=sst_processor)

    correct_count = 0
    wrong_count = 0
    failed_samples = set()
    passed_samples = set()
    error_processing_samples = set()
    for index, sample in tqdm(enumerate(list(data_samples)), desc="Processing Samples"):
        # prompt refine
        for prompt_level in range(1,6):
            gp.prompt_opt_param.base_instruction = "Classify the sentiment of the given text into one of the five categories: very negative, negative, neutral, positive, or very positive."
            gp.prompt_opt_param.specific_problem = gp.prompt_opt_param.base_instruction + "\nHere's an instance: " + sample['question']
            best_prompt = get_best_prompt(sample, prompt_level, gp)
            final_prompt = best_prompt + '\n' + output_format
            print("\n==============================Final best Prompt==============================")
            print(final_prompt)
            messages = [
                {"role": "user", "content": final_prompt}
            ]
            reasoning_content, answer_content, final_answer, completion_tokens, prompt_tokens, total_tokens = generate_with_LLM(model_name, messages,index)
            sst_processor.llm_as_judge_eval = False
            is_correct, predicted_answer = sst_processor.access_answer(final_answer, sample['final_answer'])
            print("==============================Evalutate Results==============================")
            print(is_correct)
            if is_correct:
                correct_count += 1
                write_results("sst5",sample, index,best_prompt, reasoning_content, answer_content, predicted_answer,
                              is_correct,prompt_level, completion_tokens, prompt_tokens, total_tokens,gp)
                break
            else:
                if prompt_level < 5:
                    continue
                else:
                    wrong_count += 1
                    failed_samples.add(index+1)
                    write_results("sst5",sample, index,best_prompt, reasoning_content, answer_content,
                                  predicted_answer, is_correct, prompt_level,completion_tokens, prompt_tokens, total_tokens,gp)

    print(f"\nTotal Passed: {correct_count}")
    print(f"Total Failed: {wrong_count}")
    print(f"Failed Task IDs: {', '.join(map(str, failed_samples))}")
    print(f"Error Processing Task IDs: {', '.join(map(str, error_processing_samples))}")


