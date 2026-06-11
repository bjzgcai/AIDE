# Security

## Secrets

Store credentials in a local `.env` file. `.env` and `.env.*` are ignored by
Git; `.env.example` is the only environment template intended for version
control.

Never include API keys, Hugging Face tokens, private endpoints, or proprietary
dataset contents in issues, pull requests, logs, or generated artifacts.

## Generated Code

AIDE generates and executes dataset-specific Python processors during Data
Organization. Treat generated code as untrusted until reviewed. The generated
processors run as local Python code with the permissions of the current process.

Recommended practice:

- run AIDE inside an isolated virtual environment, container, or disposable
  machine
- use bounded smoke configs before full runs
- inspect generated processor files under `outputs/<prompt-hash>/`
- avoid processing private datasets unless the execution environment is trusted

## Reporting

Please report security issues privately to the maintainers. If no private
contact channel is available yet, open a minimal public issue that describes the
affected component without exploit details or secrets, and ask for a private
coordination channel.
