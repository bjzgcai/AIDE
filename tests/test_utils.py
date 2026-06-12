import os
import tempfile
import unittest
from unittest.mock import patch

from loguru import logger

from aide.utils import (
    _llm_api_base,
    _llm_api_key,
    _llm_extra_body,
    extract_code,
    load_json,
    make_safe_call,
)


class LLMEnvironmentTests(unittest.TestCase):
    def test_llm_env_uses_single_generic_contract(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "llm-key",
                "LLM_API_BASE": "https://example.test/v1",
                "OPENAI_API_KEY": "ignored-openai-key",
                "OPENAI_API_BASE": "https://ignored.example/v1",
            },
            clear=True,
        ):
            self.assertEqual(_llm_api_key(), "llm-key")
            self.assertEqual(_llm_api_base(), "https://example.test/v1")

    def test_legacy_openai_env_is_not_used(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_API_BASE": "https://api.openai.com/v1",
            },
            clear=True,
        ):
            self.assertIsNone(_llm_api_key())
            self.assertIsNone(_llm_api_base())

    def test_llm_extra_body_json(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_EXTRA_BODY_JSON": '{"reasoning":{"enabled":false}}'},
            clear=True,
        ):
            self.assertEqual(
                _llm_extra_body(),
                {"reasoning": {"enabled": False}},
            )

    def test_llm_extra_body_json_must_be_object(self) -> None:
        with patch.dict(os.environ, {"LLM_EXTRA_BODY_JSON": "[]"}, clear=True):
            with self.assertRaises(ValueError):
                _llm_extra_body()


class JSONLoadingTests(unittest.TestCase):
    def test_missing_json_returns_none_without_error_log(self) -> None:
        messages = []
        sink_id = logger.add(
            lambda message: messages.append(str(message)),
            format="{level}:{message}",
            level="DEBUG",
        )
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self.assertIsNone(load_json(os.path.join(tmpdir, "missing.json")))
        finally:
            logger.remove(sink_id)

        self.assertFalse(any(message.startswith("ERROR:") for message in messages))


class SafeCallLoggingTests(unittest.TestCase):
    def test_make_safe_call_redacts_huggingface_tokens(self) -> None:
        secret = "hf_FAKE1234567890TOKEN"
        messages = []
        sink_id = logger.add(
            lambda message: messages.append(str(message)),
            format="{message}",
            level="WARNING",
        )

        @make_safe_call(allow_failure=True, max_tries=1)
        def always_fails(**_kwargs):
            raise RuntimeError(f"failed while using {secret}")

        try:
            self.assertIsNone(
                always_fails(
                    token=secret,
                    headers={"Authorization": f"Bearer {secret}"},
                )
            )
        finally:
            logger.remove(sink_id)

        joined = "".join(messages)
        self.assertNotIn(secret, joined)
        self.assertIn("[REDACTED]", joined)


class CodeExtractionTests(unittest.TestCase):
    def test_extract_code_from_python_fence(self) -> None:
        self.assertEqual(
            extract_code("```python\ndef process_data(data):\n    return []\n```"),
            "def process_data(data):\n    return []",
        )

    def test_extract_code_from_bare_fence(self) -> None:
        self.assertEqual(
            extract_code("```\ndef process_data(data):\n    return []\n```"),
            "def process_data(data):\n    return []",
        )

    def test_extract_code_rejects_non_code_text(self) -> None:
        self.assertEqual(extract_code("Now, please write the function."), "")

    def test_extract_code_keeps_raw_function_without_fence(self) -> None:
        code = "import json\n\ndef process_data(data):\n    return []"
        self.assertEqual(extract_code(code), code)


if __name__ == "__main__":
    unittest.main()
