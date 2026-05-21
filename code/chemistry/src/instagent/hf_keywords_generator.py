from loguru import logger
from pydantic import BaseModel

from instagent.configs import InstAgentConfig

from .utils import get_aoai_client
from .utils import OpenAIClient

system_prompt = """
## Task
Given an user query, you need to generate a list of keywords that are related to the query.
These keywords will be used to search for datasets related to the query by name in Hugging Face Hub.

## Requirements
- You need to make sure that these keywords can represent the query well, covering all aspects of the query.
- The keywords should be in the format of a list of strings, from the most relevant to the least relevant.
- The keywords should be in English and should not contain any special characters or numbers.
- Ensure you include some abbreviations or acronyms that can exist in a dataset name.
- Avoid some too general domain which may result in too many irrelevant datasets, such as "data", "research", "text", "NLP", "AI", "LLM", "training", "tuning", "optimization", "instruction", etc.
- for each keyword, you should also add its synonyms to the list. For example, if the keyword is "compound", you should also add "molecule".

## Output
You need to first output your thought process in a step by step manner. 
In the process, you need to brainstorm the keywords one by one.
After each keyword, you need to review the keywords you have generated so far, if the keyword does not fit any of the requirements, you need to remove it.
Finally, generate a list of keywords. The maximum number of keywords is 20.
"""


class KeyWords(BaseModel):
    think: str
    keywords: list[str]


class HFKeywordsGenerator:
    def __init__(self, config: InstAgentConfig):
        self.client = OpenAIClient(
            base_url=config.base_url,
            api_key=config.api_key,
            client_type="key",
        )
        self.config = config

    def generate_keywords(self, query: str) -> list[str]:
        if not self.config.use_hf_keywords_generation:
            return KeyWords(keywords=[query])
        # compose the AOIA message
        if "4o" in self.config.hf_keywords_model:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\nUser query: {query}"},
            ]
        # get the response
        resp = self.client.client.beta.chat.completions.parse(
            model=self.config.hf_keywords_model,
            messages=messages,
            response_format=KeyWords,
            temperature=self.config.hf_keywords_temperature,
            top_p=self.config.hf_keywords_top_p,
            max_tokens=self.config.hf_keywords_max_tokens,
        )
        # parse the response

        resp_message = resp.choices[0].message
        if not resp_message.parsed:
            raise ValueError(f"Failed to parse the response: {resp_message}")
        logger.info(resp_message.parsed)
        return resp_message.parsed.keywords
