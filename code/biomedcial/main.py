import hashlib
import os

from dotenv import load_dotenv
from jsonargparse import auto_cli

from src.AIDE.configs import InstAgentConfig
from src.AIDE.inst_agent import InstAgent


def main():
    load_dotenv()

    config = auto_cli(InstAgentConfig)

    # Create a hash of the prompt for the directory name
    prompt_hash = hashlib.md5(config.prompt.encode()).hexdigest()[:8]
    config.output_dir = os.path.join(config.output_dir, prompt_hash)
    inst_agent = InstAgent(config)
    inst_agent.run()


if __name__ == "__main__":
    main()
