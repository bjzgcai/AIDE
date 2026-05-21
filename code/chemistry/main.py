from jsonargparse import auto_cli

from src.instagent.configs import InstAgentConfig
from src.instagent.inst_agent import InstAgent


def main():
    config = auto_cli(InstAgentConfig)
    config.output_dir = config.output_dir + "/" + config.prompt.replace(" ", "_") + ""
    inst_agent = InstAgent(config)
    inst_agent.run()


if __name__ == "__main__":
    main()
