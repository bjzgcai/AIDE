import io
import json
import os
import re
import sys
import time

import backoff
import numpy as np
import tiktoken
from loguru import logger
from openai import AsyncOpenAI, OpenAI


def make_safe_call(allow_failure: bool = False):
    """
    A decorator that adds retry logic with backoff to API calls.

    Usage: @make_safe_call(allow_failure=True)
    """

    def decorator(func):
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=10,
            jitter=backoff.full_jitter,
            raise_on_giveup=not allow_failure,
            on_backoff=lambda details: logger.warning(
                f"Retrying {func.__name__} due to {details}"
            ),
            on_giveup=lambda details: logger.warning(
                f"Failed to call {func.__name__} due to {details}. Giving up."
            ),
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def configure_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>",
    )


def get_oai_client_async():
    base_url = os.environ.get("OPENAI_API_BASE", None)
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY environment variable is not set."
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def get_oai_client():
    base_url = os.environ.get("OPENAI_API_BASE", None)
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY environment variable is not set."
    return OpenAI(base_url=base_url, api_key=api_key)


def save_json(data, path):
    """Save data as JSON to the given path using json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """Load JSON data from the given path using json. Returns None if file is invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        return None


class OpenAIClient:
    def __init__(
        self,
        system_message: str,
    ):
        # Set the necessary variables
        self.client: OpenAI = get_oai_client()
        self.async_client: AsyncOpenAI = get_oai_client_async()
        self.system_message = system_message

    def _estimate_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Accurate token estimation using tiktoken"""
        try:
            # Determine encoding based on model name patterns
            if model.startswith("gpt-"):
                encoding_name = "cl100k_base"
            elif re.match(r"^o[1-4]-", model):
                encoding_name = "o200k_base"
            else:
                # Default fallback
                encoding_name = "cl100k_base"

            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(
                f"Failed to estimate tokens with tiktoken: {e}. Using fallback estimation."
            )
            # Fallback to character-based estimation
            return len(text) // 4

    def _truncate_messages(
        self, messages: list, model: str = "gpt-4o", max_tokens: int = 100000
    ) -> list:
        """Truncate messages to fit within token limit, keeping system message and recent messages"""
        if not messages:
            return messages

        # Calculate current token count
        total_tokens = sum(
            self._estimate_tokens(msg.get("content", ""), model) for msg in messages
        )

        if total_tokens <= max_tokens:
            return messages

        logger.warning(
            f"Messages exceed token limit ({total_tokens} > {max_tokens}). Truncating..."
        )

        # Always keep system message (first) and last user message
        truncated = []
        if messages[0].get("role") == "system":
            truncated.append(messages[0])
            remaining_messages = messages[1:]
        else:
            remaining_messages = messages

        # Keep the most recent messages that fit within limit
        current_tokens = sum(
            self._estimate_tokens(msg.get("content", ""), model) for msg in truncated
        )

        # Add messages from the end, working backwards
        for msg in reversed(remaining_messages):
            msg_tokens = self._estimate_tokens(msg.get("content", ""), model)
            if current_tokens + msg_tokens <= max_tokens:
                truncated.insert(
                    -1 if truncated and truncated[0].get("role") == "system" else 0, msg
                )
                current_tokens += msg_tokens
            else:
                break

        # If we only have system message, we need at least one more message
        if len(truncated) == 1 and remaining_messages:
            truncated.append(remaining_messages[-1])  # Add the last message

        final_tokens = sum(
            self._estimate_tokens(msg.get("content", ""), model) for msg in truncated
        )
        logger.info(
            f"Truncated from {total_tokens} to {final_tokens} tokens ({len(messages)} -> {len(truncated)} messages)"
        )

        return truncated

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(f"Retrying chat due to {details}"),
    )
    def chat(self, message: str | list, model="gpt-4o", response_format=None):
        deployment = model

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

        # Truncate messages if they exceed token limit
        message = self._truncate_messages(message, model)

        if response_format is None:
            completion = self.client.chat.completions.create(
                model=deployment, messages=message
            )
            return completion.choices[0].message.content
        else:
            response = self.client.beta.chat.completions.parse(
                model=deployment,
                messages=message,
                response_format=response_format,
            )
            ret_message = response.choices[0].message
            if ret_message.parsed:
                return ret_message.parsed
            else:
                return ret_message.content

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def async_chat(
        self,
        message: str | list,
        model="gpt-4o",
        response_format=None,
        temperature=None,
    ):
        deployment = model

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

        # Truncate messages if they exceed token limit
        message = self._truncate_messages(message, model)

        completions_kwargs = {
            "model": deployment,
            "messages": message,
        }

        if temperature is not None:
            completions_kwargs["temperature"] = temperature

        if "qwen" in model.lower():
            # No thinking
            completions_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            }

        if response_format is None:
            completion = await self.async_client.chat.completions.create(
                **completions_kwargs
            )
            return completion.choices[0].message.content
        else:
            response = await self.async_client.beta.chat.completions.parse(
                response_format=response_format, **completions_kwargs
            )
            ret_message = response.choices[0].message
            if ret_message.parsed:
                return ret_message.parsed
            else:
                return ret_message.content

    def batch_chat(
        self, batch: list[str], model: str = "gpt-4o-batch", response_format=None
    ) -> list[str]:
        """
        Submit a batch of chat prompts via Azure OpenAI batch API and return job IDs.
        :param batch: list of user prompt strings
        :param response_format: optional structured response format to enforce output
        :return: list of batch job IDs
        """

        data = []
        for i, message in enumerate(batch):
            template = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": message},
                    ],
                },
            }
            if response_format is not None:
                template["body"]["response_format"] = response_format  # type: ignore
            data.append(template)
        # Chunk and submit; collect job IDs
        job_ids: list[str] = []
        chunk_size = 10000
        batch_cnt = 0
        for start in range(0, len(data), chunk_size):
            chunk = data[start : start + chunk_size]
            jsonl_bytes = io.BytesIO(
                (
                    "\n".join(json.dumps(rec, ensure_ascii=False) for rec in chunk)
                    + "\n"
                ).encode("utf-8")
            )
            batch_file = self.client.files.create(file=jsonl_bytes, purpose="batch")
            batch_obj = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "AIDE"},
            )
            batch_cnt += 1
            job_ids.append(batch_obj.id)
            if batch_cnt % 10 == 0:
                time.sleep(1800)
        return job_ids

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
    Extract JSON from a string using json.
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


def remove_non_serializable_fields(item: dict):
    """
    Remove fields that are not serializable, e.g., embeddings or tensors.
    """
    ret = {}
    allowed_types = [str, int, float, bool]
    for k, v in item.items():
        for t in allowed_types:
            if isinstance(v, t):
                ret[k] = v
                break
    return ret


def truncate_fields(item: dict, max_length: int = 5000):
    ret = {}
    for k, v in item.items():
        if not isinstance(v, str):
            ret[k] = v
        else:
            if len(v) > max_length:
                ret[k] = v[:max_length] + "...(omitted)"
            else:
                ret[k] = v
    return ret
