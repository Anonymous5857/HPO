from typing import Dict
from ..utils.logging import get_glue_logger
import os
logger = get_glue_logger(__name__)

def call_api(messages):

    from openai import OpenAI
    from azure.identity import get_bearer_token_provider, AzureCliCredential
    from openai import AzureOpenAI

    if os.environ['USE_OPENAI_API_KEY'] == "True":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL_NAME"],
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
        return answer_content
    elif os.environ['USE_DEEPSEEK_API_KEY'] == "True":
        #===================================================================================================
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],base_url=os.environ["DEEPSEEK_BASE_URL"])
        print(f"{'=' * 20} Call {os.environ['DEEPSEEK_MODEL_NAME']} API {'=' * 20}")
        response = client.chat.completions.create(
            model=os.environ["DEEPSEEK_MODEL_NAME"],
            messages=messages,
            stream=True,
            stream_options={
                "include_usage": True
            }
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False
        print("\n" + "=" * 20 + "Thinking......" + "=" * 20 + "\n")

        for chunk in response:
            if not getattr(chunk, 'choices', None):
                print("\n" + "=" * 20 + "Token usage" + "=" * 20 + "\n")
                print(chunk.usage)
                continue

            delta = chunk.choices[0].delta

            if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
                continue

            if not getattr(delta, 'reasoning_content', None) and not is_answering:
                print("\n" + "=" * 20 + "Response......" + "=" * 20 + "\n")
                is_answering = True

            if getattr(delta, 'reasoning_content', None):
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            elif getattr(delta, 'content', None):
                print(delta.content, end='', flush=True)
                answer_content += delta.content

        return answer_content
    else:
        token_provider = get_bearer_token_provider(
                AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
            )
        client = AzureOpenAI(
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider
            )
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            messages=messages,
            temperature=0.0,
        )

    prediction = response.choices[0].message.content
    return prediction


class LLMMgr:
    @staticmethod
    def chat_completion(messages: Dict):
        try:
            return call_api(messages)
        except Exception as e:
            print(e)
            return "Sorry, I am not able to understand your query. Please try again."




