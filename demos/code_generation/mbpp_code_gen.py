import sys
sys.path.insert(0, "../../")
from hpo.promptTaskMix.promptopt.instantiate import GluePromptOpt
import os
import re
from tqdm import tqdm
from typing import Any, Dict, List, Iterable
from openai import OpenAI
import jsonlines
import yaml
from dotenv import load_dotenv
load_dotenv(override = True)
from hpo.promptTaskMix.common.human_eval.data import write_jsonl, read_problems


def update_yaml_file(file_path,config_dict):

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)


    for field,value in config_dict.items():
        data[field] = value

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("YAML file updated successfully!")


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

    output_format = "Please make sure that the final code is wrapped with <ANS_START> and <ANS_END> and does not " \
                    "contain any reasoning or related explanations."
    mbpp_samples = read_problems("mbpp.jsonl")

    path_to_config = "configs"
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")
    file_path = 'configs/promptopt_config.yaml'
    config_dict = {
        "task_description": "Your task is to complete the following code.",
        "mutation_rounds": 1,
        "few_shot_count": 3,
        "mutate_refine_iterations": 1,
    }
    update_yaml_file(file_path, config_dict)

    def get_best_prompt(sample, prompt_level, gp: GluePromptOpt):

        best_prompt = gp.get_best_prompt(sample, prompt_level=prompt_level)
        return best_prompt

    def write_results(sample, best_prompt, prompt_level, reasoning_content, answer_content, final_answer, check_result,completion_tokens,prompt_tokens,total_tokens):
        result = {
            'task_id': sample['task_id'],
            'best_prompt': best_prompt,
            'reasoning_content': reasoning_content,
            'answer_content': answer_content,
            'final_answer': final_answer,
            'passed': check_result["passed"],
            'prompt_level': prompt_level,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens
        }
        with jsonlines.open('mbpp_code_generation_with_best_prompt_results.jsonl', mode='a') as writer:
            writer.write(result)

    def generate_with_LLM(model_name, messages):
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
            answer_matches = re.findall(r"(?s)(?<=<ANS_START>)(.*?)(?=<ANS_END>)", answer_content)
            final_answer = answer_matches[0] if answer_matches else ""
            return reasoning_content, answer_content, final_answer, completion_tokens, prompt_tokens, total_tokens
        except Exception as e:
            print(f"Error processing task {sample['task_id']}: {e}")
            error_processing_samples.add(sample['task_id'])

    def evaluate_MBPP(completion: str, sample: Dict) -> Dict:
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

    gp = GluePromptOpt(promptopt_config_path,
                       setup_config_path,
                       dataset_jsonl="train_synthetic.jsonl",
                       data_processor=None)

    correct_count = 0
    wrong_count = 0
    failed_samples = set()
    passed_samples = set()
    error_processing_samples = set()

    for index, sample in tqdm(mbpp_samples.items(), desc="Processing MBPP Samples"):
        # prompt refine
        for prompt_level in range(1,6):
            best_prompt = get_best_prompt(sample, prompt_level, gp)
            final_prompt = best_prompt + '\n' + output_format
            print("\n==============================Final best Prompt==============================")
            print(final_prompt)
            messages = [
                {"role": "user", "content": final_prompt}
            ]
            reasoning_content, answer_content, final_answer, completion_tokens, prompt_tokens, total_tokens = generate_with_LLM(model_name, messages)
            check_result = evaluate_MBPP(final_answer, sample)
            print("==============================Evalutate Results==============================")
            print(check_result)

            if check_result["passed"]:
                correct_count += 1
                write_results(sample, best_prompt, prompt_level, reasoning_content, answer_content, final_answer,
                              check_result, completion_tokens, prompt_tokens, total_tokens)
                break
            else:
                if prompt_level < 5:
                    continue
                else:
                    wrong_count += 1
                    failed_samples.add(sample['task_id'])
                    write_results(sample, best_prompt, prompt_level, reasoning_content, answer_content,
                                  final_answer, check_result, completion_tokens, prompt_tokens, total_tokens)

    print(f"\nTotal Passed: {correct_count}")
    print(f"Total Failed: {wrong_count}")
    print(f"Failed Task IDs: {', '.join(map(str, failed_samples))}")
    print(f"Error Processing Task IDs: {', '.join(map(str, error_processing_samples))}")


