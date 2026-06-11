"""Command-line entry point for the full AIDE curation pipeline."""

from __future__ import annotations

import hashlib
import os

from dotenv import load_dotenv
from jsonargparse import auto_cli

from .config import AIDEConfig
from .pipeline import AIDEPipeline


def main() -> None:
    load_dotenv()
    config = auto_cli(AIDEConfig)

    prompt_hash = hashlib.md5(config.prompt.encode("utf-8")).hexdigest()[:8]
    config.output_dir = os.path.join(config.output_dir, prompt_hash)
    AIDEPipeline(config).run()


if __name__ == "__main__":
    main()
