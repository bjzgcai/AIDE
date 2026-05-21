import io
import json
import os
import re
import sys
import time

import backoff
import numpy as np
from loguru import logger
from openai import OpenAI


def configure_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>",
    )

class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        client_type: str = "key",
        resource_name: str = "",
        api_version: str = "",
        system_message: str = "",
    ):
        # Set the necessary variables
        self.resource_name = resource_name
        self.client_type = client_type
        api_version = api_version  # Replace with the appropriate API version
        if client_type == "key":
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            raise ValueError(f"Invalid client type: {client_type}")

        self.system_message = system_message

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def chat(self, message: str | list, model="openai/gpt-4o", response_format=None):
        if isinstance(message, str):
            message = [
                {
                    "role": "system",
                    "content": self.system_message,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ]

        completion = self.client.chat.completions.create(
            model=model,
            messages=message,
        )
        return completion.choices[0].message.content
        

    def embed(
        self, message: str | list[str], model: str = "text-embedding-3-large"
    ) -> np.ndarray:
        """
        Generate embeddings for one or more text messages using the specified model.
        Parameters
        ----------
        message : str or list[str]
            A single text string or a list of text strings to be converted into embeddings.
        model : str, optional
            The name of the embedding model to use (default is "text-embedding-3-large").
        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (n, d), where n is the number of input messages
            and d is the dimensionality of the embedding vector.
        Raises
        ------
        TypeError
            If `message` is not a string or a list of strings.
        ValueError
            If the embedding API returns no data or the response is malformed.
        """

        if isinstance(message, str):
            message = [message]
        ret = self.client.embeddings.create(
            model=model,
            input=message,
        )

        ret_embs = [r.embedding for r in ret.data]
        ret_embs = np.array(ret_embs)
        return ret_embs

    async def async_embed(
        self, message: str | list[str], model: str = "text-embedding-3-large"
    ) -> np.ndarray:
        """
        Asynchronous version of the embed method.
        """
        if isinstance(message, str):
            message = [message]
        ret = await self.async_client.embeddings.create(
            model=model,
            input=message,
        )

        ret_embs = [r.embedding for r in ret.data]
        ret_embs = np.array(ret_embs)
        return ret_embs



def extract_json(s):
    """
    Extract JSON from a string.
    """
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, s, re.DOTALL)
    if matches == []:
        return json.loads(s)
    return json.loads(matches[0].strip())


def extract_code(s: str) -> str:
    """
    Extract Python code from a string.
    """
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, s, re.DOTALL)
    if matches == []:
        return s
    return matches[0]
