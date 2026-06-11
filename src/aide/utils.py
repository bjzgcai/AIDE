import io
import json
import os
import re
import sys
import time
from collections.abc import Callable

import backoff
import numpy as np
import tiktoken
from loguru import logger
from openai import AsyncOpenAI, OpenAI

from .metrics import get_active_metrics_tracker


def make_safe_call(
    allow_failure: bool = False,
    giveup: Callable[[Exception], bool] | None = None,
    max_tries: int = 10,
):
    """
    A decorator that adds retry logic with backoff to API calls.

    Usage: @make_safe_call(allow_failure=True)
    Use giveup to mark exceptions as non-retryable.
    """

    def decorator(func):
        should_give_up = giveup or (lambda exc: False)

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_tries,
            jitter=backoff.full_jitter,
            giveup=should_give_up,
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


def _llm_api_base() -> str | None:
    return os.environ.get("LLM_API_BASE")


def _llm_api_key() -> str | None:
    return os.environ.get("LLM_API_KEY")


def _llm_extra_body() -> dict | None:
    raw = os.environ.get("LLM_EXTRA_BODY_JSON")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM_EXTRA_BODY_JSON must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM_EXTRA_BODY_JSON must decode to a JSON object.")
    return value


def get_oai_client_async():
    api_key = _llm_api_key()
    assert api_key, "Set LLM_API_KEY."
    return AsyncOpenAI(base_url=_llm_api_base(), api_key=api_key)


def get_oai_client():
    api_key = _llm_api_key()
    assert api_key, "Set LLM_API_KEY."
    return OpenAI(base_url=_llm_api_base(), api_key=api_key)


def save_json(data, path):
    """Save data as JSON to the given path using an atomic replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_json(path):
    """Load JSON data from the given path using json. Returns None if file is invalid."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        return None


class OpenAIClient:
    def __init__(
        self,
        system_message: str,
    ):
        self._sync_clients: dict[str, OpenAI] = {}
        self._async_clients: dict[str, AsyncOpenAI] = {}
        self.system_message = system_message

    def _client_config_for_model(self, model: str) -> tuple[str | None, str]:
        base_url = _llm_api_base()
        api_key = _llm_api_key()
        if not api_key:
            raise RuntimeError(
                f"No API key configured for model={model}. Set LLM_API_KEY."
            )
        return base_url, api_key

    def _get_sync_client_for_model(self, model: str) -> OpenAI:
        base_url, api_key = self._client_config_for_model(model)
        cache_key = base_url or ""
        if cache_key not in self._sync_clients:
            self._sync_clients[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
        return self._sync_clients[cache_key]

    def _get_async_client_for_model(self, model: str) -> AsyncOpenAI:
        base_url, api_key = self._client_config_for_model(model)
        cache_key = base_url or ""
        if cache_key not in self._async_clients:
            self._async_clients[cache_key] = AsyncOpenAI(
                base_url=base_url, api_key=api_key
            )
        return self._async_clients[cache_key]

    def _estimate_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Accurate token estimation using tiktoken"""
        try:
            if model.startswith("gpt-"):
                encoding_name = "cl100k_base"
            elif re.match(r"^o[1-4]-", model):
                encoding_name = "o200k_base"
            else:
                encoding_name = "cl100k_base"

            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(
                f"Failed to estimate tokens with tiktoken: {e}. Using fallback estimation."
            )
            return len(text) // 4

    def _stringify_content_for_tokens(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if hasattr(value, "model_dump_json"):
            try:
                return value.model_dump_json()
            except Exception:
                pass
        if hasattr(value, "model_dump"):
            try:
                return json.dumps(value.model_dump(), ensure_ascii=False)
            except Exception:
                pass
        return str(value)

    def _estimate_message_tokens(self, messages: list, model: str) -> int:
        total = 0
        for message in messages:
            total += self._estimate_tokens(str(message.get("role", "")), model)
            total += self._estimate_tokens(
                self._stringify_content_for_tokens(message.get("content", "")), model
            )
        return total

    def _record_chat_metrics(
        self,
        *,
        model: str,
        api_time_sec: float,
        response_obj,
        message: list,
        returned_value,
    ) -> None:
        tracker = get_active_metrics_tracker()
        if tracker is None:
            return

        prompt_tokens = self._estimate_message_tokens(message, model)
        completion_tokens = self._estimate_tokens(
            self._stringify_content_for_tokens(returned_value), model
        )
        total_tokens = prompt_tokens + completion_tokens
        usage = getattr(response_obj, "usage", None)

        tracker.record_llm_usage(
            model=model,
            api_time_sec=api_time_sec,
            usage=usage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _record_chat_failure(self, *, model: str, api_time_sec: float) -> None:
        tracker = get_active_metrics_tracker()
        if tracker is None:
            return
        tracker.record_llm_failure(model=model, api_time_sec=api_time_sec)

    def _truncate_messages(
        self, messages: list, model: str = "gpt-4o", max_tokens: int = 100000
    ) -> list:
        """Truncate messages to fit within token limit, keeping system message and recent messages"""
        if not messages:
            return messages

        total_tokens = sum(
            self._estimate_tokens(
                self._stringify_content_for_tokens(msg.get("content", "")), model
            )
            for msg in messages
        )

        if total_tokens <= max_tokens:
            return messages

        logger.warning(
            f"Messages exceed token limit ({total_tokens} > {max_tokens}). Truncating..."
        )

        truncated = []
        if messages[0].get("role") == "system":
            truncated.append(messages[0])
            remaining_messages = messages[1:]
        else:
            remaining_messages = messages

        current_tokens = sum(
            self._estimate_tokens(
                self._stringify_content_for_tokens(msg.get("content", "")), model
            )
            for msg in truncated
        )

        for msg in reversed(remaining_messages):
            msg_tokens = self._estimate_tokens(
                self._stringify_content_for_tokens(msg.get("content", "")), model
            )
            if current_tokens + msg_tokens <= max_tokens:
                truncated.insert(
                    -1 if truncated and truncated[0].get("role") == "system" else 0,
                    msg,
                )
                current_tokens += msg_tokens
            else:
                break

        if len(truncated) == 1 and remaining_messages:
            truncated.append(remaining_messages[-1])

        final_tokens = sum(
            self._estimate_tokens(
                self._stringify_content_for_tokens(msg.get("content", "")), model
            )
            for msg in truncated
        )
        if final_tokens > max_tokens:
            for idx in range(len(truncated) - 1, -1, -1):
                content = truncated[idx].get("content", "")
                if not isinstance(content, str):
                    continue
                other_tokens = final_tokens - self._estimate_tokens(content, model)
                remaining_budget = max_tokens - other_tokens
                if remaining_budget <= 0:
                    continue
                truncated[idx] = {
                    **truncated[idx],
                    "content": self._truncate_text_to_tokens(
                        content, model=model, max_tokens=remaining_budget
                    ),
                }
                break
            final_tokens = sum(
                self._estimate_tokens(
                    self._stringify_content_for_tokens(msg.get("content", "")), model
                )
                for msg in truncated
            )
        logger.info(
            f"Truncated from {total_tokens} to {final_tokens} tokens ({len(messages)} -> {len(truncated)} messages)"
        )

        return truncated

    def _truncate_text_to_tokens(
        self, text: str, model: str = "gpt-4o", max_tokens: int = 100000
    ) -> str:
        if not text:
            return text

        current_tokens = self._estimate_tokens(text, model)
        if current_tokens <= max_tokens:
            return text

        suffix = "\n...(truncated)"
        suffix_tokens = self._estimate_tokens(suffix, model)
        if max_tokens <= suffix_tokens:
            return suffix[: max(0, len(suffix) // 2)]

        target_tokens = max_tokens - suffix_tokens
        ratio = target_tokens / max(current_tokens, 1)
        candidate = text[: max(1, int(len(text) * ratio))] + suffix

        while self._estimate_tokens(candidate, model) > max_tokens and len(
            candidate
        ) > len(suffix):
            candidate = (
                candidate[: max(len(suffix) + 1, int(len(candidate) * 0.8))] + suffix
            )

        return candidate

    def _preview_for_logs(self, value, max_chars: int = 400) -> str:
        if value is None:
            return "<None>"
        text = value if isinstance(value, str) else repr(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...(truncated)"

    def _messages_preview_for_logs(
        self, message: list[dict], max_chars: int = 400
    ) -> list[dict[str, str]]:
        return [
            {
                "role": str(msg.get("role", "")),
                "content": self._preview_for_logs(
                    msg.get("content", ""), max_chars=max_chars
                ),
            }
            for msg in message
        ]

    def _response_preview_for_logs(self, response_obj, max_chars: int = 1500) -> str:
        if response_obj is None:
            return "<None>"
        try:
            if hasattr(response_obj, "model_dump"):
                dumped = json.dumps(response_obj.model_dump(), ensure_ascii=False)
            else:
                dumped = repr(response_obj)
        except Exception as exc:
            dumped = f"<response serialization failed: {exc.__class__.__name__}: {exc}>"
        return self._preview_for_logs(dumped, max_chars=max_chars)

    def _raise_malformed_chat_response(
        self,
        *,
        stage: str,
        model: str,
        message: list[dict],
        response_obj,
        client_base_url: str,
    ) -> None:
        msg = (
            f"Malformed chat response at stage={stage}. "
            f"model={model}, base_url={client_base_url or '<empty>'}. "
            f"message_preview={self._messages_preview_for_logs(message)}. "
            f"response_preview={self._response_preview_for_logs(response_obj)}"
        )
        raise ValueError(msg)

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

        message = self._truncate_messages(message, model)
        client = self._get_sync_client_for_model(deployment)
        completions_kwargs = {
            "model": deployment,
            "messages": message,
        }
        extra_body = _llm_extra_body()
        if extra_body is not None:
            completions_kwargs["extra_body"] = extra_body

        request_start = time.perf_counter()
        if response_format is None:
            try:
                completion = client.chat.completions.create(**completions_kwargs)
                response_text = completion.choices[0].message.content
            except Exception:
                self._record_chat_failure(
                    model=deployment,
                    api_time_sec=time.perf_counter() - request_start,
                )
                raise
            self._record_chat_metrics(
                model=deployment,
                api_time_sec=time.perf_counter() - request_start,
                response_obj=completion,
                message=message,
                returned_value=response_text,
            )
            return response_text

        try:
            response = client.beta.chat.completions.parse(
                response_format=response_format,
                **completions_kwargs,
            )
            ret_message = response.choices[0].message
            returned_value = (
                ret_message.parsed if ret_message.parsed else ret_message.content
            )
        except Exception:
            self._record_chat_failure(
                model=deployment,
                api_time_sec=time.perf_counter() - request_start,
            )
            raise
        self._record_chat_metrics(
            model=deployment,
            api_time_sec=time.perf_counter() - request_start,
            response_obj=response,
            message=message,
            returned_value=returned_value,
        )
        return returned_value

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

        message = self._truncate_messages(message, model)
        async_client = self._get_async_client_for_model(deployment)
        client_base_url = _llm_api_base() or ""

        completions_kwargs = {
            "model": deployment,
            "messages": message,
        }
        extra_body = _llm_extra_body()
        if extra_body is not None:
            completions_kwargs["extra_body"] = extra_body

        if temperature is not None:
            completions_kwargs["temperature"] = temperature

        request_log_context = {
            "model": deployment,
            "temperature": temperature,
            "response_format": bool(response_format),
            "base_url": client_base_url,
            "messages_preview": self._messages_preview_for_logs(message),
            "extra_body": completions_kwargs.get("extra_body"),
        }

        request_start = time.perf_counter()
        if response_format is None:
            try:
                completion = await async_client.chat.completions.create(
                    **completions_kwargs
                )
            except Exception as e:
                self._record_chat_failure(
                    model=deployment,
                    api_time_sec=time.perf_counter() - request_start,
                )
                logger.exception(
                    "async_chat provider call failed before response parsing. "
                    f"exception={e.__class__.__name__}: {e}. "
                    f"context={json.dumps(request_log_context, ensure_ascii=False, default=str)}"
                )
                raise
            choices = getattr(completion, "choices", None)
            if not choices:
                self._record_chat_failure(
                    model=deployment,
                    api_time_sec=time.perf_counter() - request_start,
                )
                self._raise_malformed_chat_response(
                    stage="choices_missing_or_empty",
                    model=deployment,
                    message=message,
                    response_obj=completion,
                    client_base_url=client_base_url,
                )
            first_choice = choices[0]
            if first_choice is None:
                self._record_chat_failure(
                    model=deployment,
                    api_time_sec=time.perf_counter() - request_start,
                )
                self._raise_malformed_chat_response(
                    stage="first_choice_is_none",
                    model=deployment,
                    message=message,
                    response_obj=completion,
                    client_base_url=client_base_url,
                )
            first_message = getattr(first_choice, "message", None)
            if first_message is None:
                self._record_chat_failure(
                    model=deployment,
                    api_time_sec=time.perf_counter() - request_start,
                )
                self._raise_malformed_chat_response(
                    stage="choice_message_is_none",
                    model=deployment,
                    message=message,
                    response_obj=completion,
                    client_base_url=client_base_url,
                )
            content = getattr(first_message, "content", None)
            if content is None:
                self._record_chat_failure(
                    model=deployment,
                    api_time_sec=time.perf_counter() - request_start,
                )
                self._raise_malformed_chat_response(
                    stage="choice_message_content_is_none",
                    model=deployment,
                    message=message,
                    response_obj=completion,
                    client_base_url=client_base_url,
                )
            self._record_chat_metrics(
                model=deployment,
                api_time_sec=time.perf_counter() - request_start,
                response_obj=completion,
                message=message,
                returned_value=content,
            )
            return content

        try:
            response = await async_client.beta.chat.completions.parse(
                response_format=response_format, **completions_kwargs
            )
        except Exception as e:
            self._record_chat_failure(
                model=deployment,
                api_time_sec=time.perf_counter() - request_start,
            )
            logger.exception(
                "async_chat parse call failed before response parsing. "
                f"exception={e.__class__.__name__}: {e}. "
                f"context={json.dumps(request_log_context, ensure_ascii=False, default=str)}"
            )
            raise
        choices = getattr(response, "choices", None)
        if not choices:
            self._record_chat_failure(
                model=deployment,
                api_time_sec=time.perf_counter() - request_start,
            )
            self._raise_malformed_chat_response(
                stage="parse_choices_missing_or_empty",
                model=deployment,
                message=message,
                response_obj=response,
                client_base_url=client_base_url,
            )
        ret_message = getattr(choices[0], "message", None)
        if ret_message is None:
            self._record_chat_failure(
                model=deployment,
                api_time_sec=time.perf_counter() - request_start,
            )
            self._raise_malformed_chat_response(
                stage="parse_choice_message_is_none",
                model=deployment,
                message=message,
                response_obj=response,
                client_base_url=client_base_url,
            )
        returned_value = (
            ret_message.parsed if ret_message.parsed else ret_message.content
        )
        self._record_chat_metrics(
            model=deployment,
            api_time_sec=time.perf_counter() - request_start,
            response_obj=response,
            message=message,
            returned_value=returned_value,
        )
        return returned_value

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
        job_ids: list[str] = []
        chunk_size = 10000
        batch_cnt = 0
        client = self._get_sync_client_for_model(model)
        for start in range(0, len(data), chunk_size):
            chunk = data[start : start + chunk_size]
            jsonl_bytes = io.BytesIO(
                (
                    "\n".join(json.dumps(rec, ensure_ascii=False) for rec in chunk)
                    + "\n"
                ).encode("utf-8")
            )
            batch_file = client.files.create(file=jsonl_bytes, purpose="batch")
            batch_obj = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "aide-curator"},
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
        client = self._get_sync_client_for_model(model)
        ret = client.embeddings.create(
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
        async_client = self._get_async_client_for_model(model)
        ret = await async_client.embeddings.create(
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


def sanitize_item_for_prompt(
    item: dict,
    *,
    max_depth: int = 3,
    max_str_length: int = 4000,
    max_list_items: int = 20,
):
    """
    Keep compact, prompt-friendly JSON-like content while dropping embedding-like arrays.
    """

    def _sanitize(value, depth: int):
        if depth < 0:
            return None
        if isinstance(value, str):
            if len(value) > max_str_length:
                return value[:max_str_length] + "...(omitted)"
            return value
        if isinstance(value, int | float | bool) or value is None:
            return value
        if isinstance(value, dict):
            sanitized = {}
            for key, nested_value in value.items():
                if not isinstance(key, str):
                    continue
                nested = _sanitize(nested_value, depth - 1)
                if nested not in (None, {}, []):
                    sanitized[key] = nested
            return sanitized
        if isinstance(value, list | tuple):
            if not value:
                return []
            if all(isinstance(v, int | float | bool) or v is None for v in value):
                return (
                    list(value[:max_list_items])
                    if len(value) <= max_list_items
                    else None
                )

            sanitized = []
            for nested_value in value[:max_list_items]:
                nested = _sanitize(nested_value, depth - 1)
                if nested is not None:
                    sanitized.append(nested)
            if len(value) > max_list_items:
                sanitized.append(
                    f"...({len(value) - max_list_items} more items omitted)"
                )
            return sanitized
        return None

    sanitized_item = _sanitize(item, max_depth)
    return sanitized_item if isinstance(sanitized_item, dict) else {}
