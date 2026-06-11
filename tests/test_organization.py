import unittest

from aide import AIDEConfig, DatasetInfo
from aide.organization_instruction import InstructionOrganizer
from aide.organization_rag import RAGOrganizer


class FakeChatClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages = []

    def chat(self, message, model=None, response_format=None):
        self.messages.append(message)
        if not self.responses:
            raise AssertionError("FakeChatClient received more calls than expected")
        return self.responses.pop(0)


def _dataset_info() -> DatasetInfo:
    return DatasetInfo(
        name="demo/science",
        subset="default",
        number_of_rows=2,
        columns=["question", "answer"],
        description="Toy scientific QA dataset",
        hf_tags=["science"],
        modalities=["text"],
        license="mit",
        n_likes=1,
        n_downloads_last_month=1,
    )


class InstructionOrganizationTests(unittest.TestCase):
    def test_invalid_task_columns_are_rejected(self) -> None:
        organizer = InstructionOrganizer(AIDEConfig(prompt="science"))
        task = {
            "task": "qa",
            "input columns": ["question"],
            "output columns": ["missing_answer"],
        }

        self.assertFalse(organizer._is_valid_task(task, _dataset_info()))

    def test_generated_code_uses_isolated_namespace(self) -> None:
        organizer = InstructionOrganizer(AIDEConfig(prompt="science"))
        fake_client = FakeChatClient(["not_process_data = lambda rows: []"])

        result = organizer._generate_process_data_function(
            fake_client, [{"role": "system", "content": "generate code"}]
        )

        self.assertIsNone(result.process_data)

    def test_validation_failure_retries_with_revised_code(self) -> None:
        config = AIDEConfig(
            prompt="science",
            instruction_quality_check=False,
            instruction_max_retries=2,
        )
        organizer = InstructionOrganizer(config)
        fake_client = FakeChatClient(
            [
                "```python\ndef process_data(rows):\n    return ['malformed']\n```",
                (
                    "```python\n"
                    "def process_data(rows):\n"
                    "    return [row['question'] + '\\t' + row['answer'] for row in rows]\n"
                    "```"
                ),
            ]
        )

        processor = organizer._build_validated_processor(
            gpt4=fake_client,
            chat_history=[{"role": "system", "content": "generate code"}],
            sample_rows=[{"question": "What is H2O?", "answer": "Water"}],
            prompts_list={"need_template": False},
            task={
                "task": "qa",
                "input columns": ["question"],
                "output columns": ["answer"],
            },
        )

        self.assertIsNotNone(processor)
        self.assertEqual(len(fake_client.messages), 2)
        self.assertEqual(
            processor.process_data(
                [{"question": "Atomic number 6?", "answer": "Carbon"}]
            ),
            ["Atomic number 6?\tCarbon"],
        )


class RAGOrganizationTests(unittest.TestCase):
    def test_generated_code_uses_isolated_namespace(self) -> None:
        organizer = RAGOrganizer(AIDEConfig(prompt="science"))
        fake_client = FakeChatClient(["not_process_data = lambda row: []"])

        process_data, _, _, _ = organizer._generate_process_data_function(
            fake_client,
            [{"role": "system", "content": "generate code"}],
        )

        self.assertIsNone(process_data)


if __name__ == "__main__":
    unittest.main()
