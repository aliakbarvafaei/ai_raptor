import logging
import os

from openai import OpenAI


import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass

import requests
import json
from typing import Dict, Any

class OllamaBaseModel:
    def __init__(self, model_name: str, ollama_url: str = "http://localhost:11434/api"):
        """
        :param model_name: Name of the local Ollama model.
        :param ollama_url: URL to Ollama's local API.
        """
        self.model_name = model_name
        self.ollama_url = ollama_url

    def _post_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a POST request to the Ollama API and returns the response."""
        url = f"{self.ollama_url}/{endpoint}"
        try:
            response = requests.post(
                url, headers={"Content-Type": "application/json"}, data=json.dumps(payload)
            )
            response.raise_for_status()
            
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request to {url} failed: {str(e)}")


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]
    


class OllamaQAModel(BaseQAModel):

    def __init__(self, model="dolphin-phi"):

        """

        Initializes the Ollama model with the specified model version.

        Args:

            model (str, optional): The Ollama model version to use. Defaults to "dolphin-phi".

        """

        self.model = model




    def answer_question(self, context, question):

        """

        Generates an answer to the given question using the Ollama model.

        Args:

            context (str): The context or background information for the question.

            question (str): The question to generate an answer for.

        Returns:

            str: The generated answer.

        """

        import ollama

        response = ollama.chat(model=self.model, messages=[

                {"role": "system", "content": "You are Question Answering Portal"},

                {

                    "role": "user",

                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",

                },

            ])

        return response['message']['content']
    

import os
from tenacity import retry, wait_random_exponential, stop_after_attempt

class DeepSeekQAModel(BaseQAModel):
    def __init__(self, model="deepseek-chat"):
        """
        Initializes the DeepSeek model with the specified model version.

        Args:
            model (str, optional): The DeepSeek model version to use for generating answers. Defaults to "deepseek-model-v1".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates an answer to the given question based on the provided context using the DeepSeek model.

        Args:
            context (str): The context to base the answer on.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated answer. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop generation. Defaults to None.

        Returns:
            str: The generated answer.
        """
        response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
        return response.choices[0].text.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Attempts to answer the question using the DeepSeek model with retry logic.

        Args:
            context (str): The context to base the answer on.
            question (str): The question to answer.
            max_tokens (int, optional): The maximum number of tokens in the generated answer. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop generation. Defaults to None.

        Returns:
            str: The generated answer or an error message if an exception occurs.
        """
        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return str(e)
        
class OllamaQAModelV2(BaseQAModel, OllamaBaseModel):
    def answer_question(self, context: str, question: str) -> str:
        """Answer the question based on the given context using the local Ollama model."""
        payload = {
            "model": self.model_name,
            "prompt": f"Given the context: {context}, answer the question: {question}",
            "max_tokens": 256,
        }
        result = self._post_request("generate", payload)
        # Parse the raw responses into a list of JSON objects
        responses = [json.loads(line.strip()) for line in result.text.strip().split("\n")]

        # Combine the `response` values
        combined_response = "".join([resp["response"] for resp in responses])

        return combined_response