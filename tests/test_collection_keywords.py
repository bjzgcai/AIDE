import unittest

from aide import AIDEConfig
from aide.collection_keywords import KeywordGenerator


class KeywordGeneratorJSONExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = KeywordGenerator(AIDEConfig(prompt="demo"))

    def test_extracts_raw_json_array(self) -> None:
        raw = '["medical question answering", "clinical qa"]'

        json_str = self.generator._extract_json_from_response(raw, attempt_num=1)

        self.assertEqual(raw, json_str)
        self.assertEqual(
            ["medical question answering", "clinical qa"],
            self.generator._parse_and_validate_keywords(json_str, attempt_num=1),
        )

    def test_extracts_json_fence(self) -> None:
        response = '```json\n["medical qa"]\n```'

        self.assertEqual(
            '["medical qa"]',
            self.generator._extract_json_from_response(response, attempt_num=1),
        )

    def test_extracts_bare_fence(self) -> None:
        response = '```\n["clinical qa"]\n```'

        self.assertEqual(
            '["clinical qa"]',
            self.generator._extract_json_from_response(response, attempt_num=1),
        )

    def test_extracts_embedded_json_array(self) -> None:
        response = 'Here are the keywords:\n["biomedical qa", "healthcare qa"]'

        self.assertEqual(
            '["biomedical qa", "healthcare qa"]',
            self.generator._extract_json_from_response(response, attempt_num=1),
        )


if __name__ == "__main__":
    unittest.main()
