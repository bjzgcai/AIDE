import json
import re

import backoff
from loguru import logger

from .config import AIDEConfig
from .prompts import prompt_for_keywords_generation
from .utils import OpenAIClient


class KeywordGenerator:
    def __init__(self, config: AIDEConfig):
        self.client = OpenAIClient(
            system_message=prompt_for_keywords_generation,
        )
        self.config = config

    def _extract_json_from_response(
        self, response: str, attempt_num: int
    ) -> str | None:
        """Extract parseable JSON content from an LLM response."""
        candidates = [
            match.strip()
            for match in re.findall(
                r"```(?:\w+)?\s*(.*?)```",
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )
        ]
        stripped_response = response.strip()
        if stripped_response:
            candidates.append(stripped_response)

        decoder = json.JSONDecoder()
        for start, char in enumerate(response):
            if char not in "[{":
                continue
            try:
                _, end = decoder.raw_decode(response[start:])
            except json.JSONDecodeError:
                continue
            candidates.append(response[start : start + end].strip())

        for candidate in candidates:
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return candidate

        logger.warning(f"No parseable JSON found in response for attempt {attempt_num}")
        return None

    def _parse_and_validate_keywords(
        self, json_str: str, attempt_num: int
    ) -> list[str] | None:
        """Parse JSON string and validate that it contains a list of string keywords."""
        try:
            keywords = json.loads(json_str)
            logger.info(f"Parsed keywords from attempt {attempt_num}: {keywords}")

            if not isinstance(keywords, list):
                logger.warning(
                    f"Parsed keywords is not a list for attempt {attempt_num}"
                )
                return None

            if not all(isinstance(k, str) for k in keywords):
                logger.warning(
                    f"Not all keywords are strings for attempt {attempt_num}"
                )
                return None

            return keywords

        except Exception as e:
            logger.warning(f"Failed to parse JSON for attempt {attempt_num}: {e}")
            return None

    def _generate_keywords_single_attempt(
        self, query: str, attempt_num: int
    ) -> list[str] | None:
        """Generate keywords for a single attempt."""
        try:
            resp = self.client.chat(
                message=query,
                model=self.config.collection_keyword_model,
            )

            if resp is None:
                logger.warning(f"No response received for attempt {attempt_num}")
                return None

            logger.info(
                f"Response from collection keyword generation (attempt {attempt_num}): {resp}"
            )

            # Extract JSON from response
            json_str = self._extract_json_from_response(resp, attempt_num)
            if json_str is None:
                return None

            # Parse and validate keywords
            return self._parse_and_validate_keywords(json_str, attempt_num)

        except Exception as e:
            logger.warning(
                f"Failed to generate keywords for attempt {attempt_num}: {e}"
            )
            return None

    def _collect_all_keywords(self, query: str) -> set[str]:
        """Collect keywords from multiple generation attempts."""
        all_keywords = set()

        for attempt in range(self.config.collection_keyword_attempts):
            logger.info(f"Attempt {attempt}...")
            keywords = self._generate_keywords_single_attempt(query, attempt + 1)
            if keywords:
                all_keywords.update(keywords)

        return all_keywords

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    def generate_keywords(self, query: str) -> list[str]:
        """Generate keywords for the given query using multiple attempts."""
        if not self.config.use_llm_collection_keywords:
            return [query]

        logger.info("Generating collection keywords")

        # Collect keywords from all attempts
        all_keywords = self._collect_all_keywords(query)

        # Convert set back to list
        final_keywords = list(all_keywords)
        logger.info(f"Final union of all keywords: {final_keywords}")

        # If no keywords were successfully generated, fall back to original query
        if not final_keywords:
            logger.warning(
                "No keywords were successfully generated, falling back to original query"
            )
            return [query]

        return final_keywords
