import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
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


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e
class OllamaSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="dolphin-phi"):
        """
        Initializes the Ollama model with the specified model version.
        Args:
            model (str, optional): The Ollama model version to use. Defaults to "dolphin-phi".
        """
        self.model = model

    def summarize(self, context):
        import ollama
        try:
            response = ollama.chat(model=self.model, messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ])
            return response['message']['content']
        except Exception as e:
            print(e)
            return e
  
class OllamaSummarizationModelV2(BaseSummarizationModel, OllamaBaseModel):
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """Summarize the given context using the local Ollama model."""
        payload = {
            "model": self.model_name,
            "prompt": f"Write a detailed summary of the following text: {context}",
            "max_tokens": max_tokens,
        }
        result = self._post_request("generate", payload)
        # Parse the raw responses into a list of JSON objects
        responses = [json.loads(line.strip()) for line in result.text.strip().split("\n")]

        # Combine the `response` values
        combined_response = "".join([resp["response"] for resp in responses])

        return combined_response