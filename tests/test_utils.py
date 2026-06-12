import os
import unittest
from unittest.mock import patch

from aide.utils import _llm_api_base, _llm_api_key, _llm_extra_body, extract_code


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
