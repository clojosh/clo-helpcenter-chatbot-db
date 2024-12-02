import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .misc import trim_tokens

backend_dir = Path(__file__).parent.parent.parent

sys.path.append(str(backend_dir))
from tools.misc import num_tokens_from_string

# https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=python-secure%2Cglobal-standard%2Cstandard-chat-completions#gpt-4o-and-gpt-4-turbo
GPT_4_MINI_MAX_INPUT_TOKENS = 128000
GPT_4_MINI_MAX_OUTPUT_TOKENS = 16000
EMBEDDING_ADA_002_MAX_INPUT_TOKENS = 8191


class OpenAIHelper:
    def __init__(
        self,
        openai_client: AzureOpenAI,
        AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        AZURE_OPENAI_EMB_DEPLOYMENT,
        language="English",
    ):
        self.openai_client = openai_client
        self.AZURE_OPENAI_CHATGPT_DEPLOYMENT = AZURE_OPENAI_CHATGPT_DEPLOYMENT
        self.AZURE_OPENAI_EMB_DEPLOYMENT = AZURE_OPENAI_EMB_DEPLOYMENT
        self.language = language

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_embeddings(self, text: str) -> list[float]:
        """
        Generate embeddings for a given text.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            list[float]: The generated embeddings.

        Raises:
            openai.error.OpenAIError: If the request to the OpenAI API fails.
        """
        tokens = num_tokens_from_string(text, "text-embedding-ada-002")

        if tokens >= EMBEDDING_ADA_002_MAX_INPUT_TOKENS:
            text = text[:EMBEDDING_ADA_002_MAX_INPUT_TOKENS]

        return (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.AZURE_OPENAI_EMB_DEPLOYMENT,
            )
            .data[0]
            .embedding
        )

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_pdf_summary(self, pdf):
        """
        Summarize a PDF

        Args:
            pdf (str): The text of the PDF to summarize

        Returns:
            str: A summary of the PDF
        """
        # If the PDF is too long, trim it to a length that OpenAI can handle
        tokens = num_tokens_from_string(pdf, "gpt-4o-mini")

        if tokens >= GPT_4_MINI_MAX_INPUT_TOKENS:
            pdf = pdf[:GPT_4_MINI_MAX_INPUT_TOKENS]

        # Create the prompt for the AI
        messages = [
            {
                "role": "user",
                "content": f"Provide a comprehensive guide of the given text. Include all step-by-step instructions, definitions, and warranties. {pdf}",
            }
        ]

        # Ask the AI to generate a summary
        chat_completion = self.openai_client.chat.completions.create(
            model=self.AZURE_OPENAI_CHATGPT_DEPLOYMENT, messages=messages, temperature=0.7, max_tokens=1000, n=1
        )

        # Extract the summary from the response
        summary = chat_completion.choices[0].message.content
        summary = re.sub(r"\n+", " ", summary)
        summary = re.sub(r"\s+", " ", summary)

        return summary
