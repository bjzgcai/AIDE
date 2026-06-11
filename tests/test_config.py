import unittest

from aide import AIDEConfig


class ConfigTests(unittest.TestCase):
    def test_config_defaults_are_release_safe(self) -> None:
        config = AIDEConfig(prompt="demo")

        self.assertEqual(config.cache_dir, "cache/huggingface")
        self.assertEqual(config.output_dir, "outputs")
        self.assertEqual(config.organization_format, "instruction")
        self.assertFalse(config.use_llm_collection_keywords)
        self.assertEqual(config.collection_keyword_model, "openai/gpt-4o-mini")
        self.assertEqual(config.collection_keyword_attempts, 3)
        self.assertEqual(config.selection_model, "openai/gpt-4o-mini")
        self.assertEqual(config.selection_score_threshold, 5)
        self.assertEqual(config.organization_model, "openai/gpt-4o-mini")
        self.assertEqual(config.instruction_max_retries, 3)
        self.assertEqual(config.instruction_sample_size, 5)
        self.assertTrue(config.instruction_quality_check)
        self.assertEqual(config.instruction_max_items_per_dataset, 0)
        self.assertEqual(config.rag_sample_size, 15)
        self.assertEqual(config.rag_max_retries, 3)
        self.assertTrue(config.rag_streaming)


if __name__ == "__main__":
    unittest.main()
