import json

import backoff
from loguru import logger

from AIDE.configs import InstAgentConfig

from .prompts import prompt_for_keywords_generation
from .utils import OpenAIClient


class HFKeywordsGenerator:
    def __init__(self, config: InstAgentConfig):
        self.client = OpenAIClient(
            system_message=prompt_for_keywords_generation,
        )
        self.config = config

    def _extract_json_from_response(
        self, response: str, attempt_num: int
    ) -> str | None:
        """Extract JSON content from markdown code blocks in the response."""
        # Handle both ```json\n[...]``` and ```[...]``` code blocks
        # Try to find a code block containing JSON, but fallback to any code block if not found
        start = response.find("```json")
        if start != -1:
            end = response.find("```", start + 7)
            if end != -1:
                return response[start + 7 : end].strip()
        # If not found, try to find any code block (e.g., ```\n[...]```)
        start = response.find("```")
        if start != -1:
            end = response.find("```", start + 3)
            if end != -1:
                return response[start + 3 : end].strip()
        logger.warning(f"No JSON found in response for attempt {attempt_num}")
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
                model=self.config.hf_keywords_model,
            )

            if resp is None:
                logger.warning(f"No response received for attempt {attempt_num}")
                return None

            logger.info(
                f"Response from HF Keywords Generation (attempt {attempt_num}): {resp}"
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

        for attempt in range(self.config.hf_keywords_generation_attempts):
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
        if not self.config.use_hf_keywords_generation:
            return [query]

        logger.info("Generating key words")

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


if __name__ == "__main__":
    config = InstAgentConfig(prompt="bio", use_hf_keywords_generation=True)
    generator = HFKeywordsGenerator(config)
    keywords = generator.generate_keywords("bio")
    print(f"Generated keywords: {keywords}")
