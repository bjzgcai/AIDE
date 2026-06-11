# Contributing

Thanks for your interest in improving AIDE. This release focuses on the
data-curation platform described in the paper: Data Collection, Data Selection,
and Data Organization.

## Development Setup

```bash
uv sync --extra dev
```

Run the local checks before opening a pull request:

```bash
uv run --extra dev ruff check .
uv run python -m unittest discover -s tests
uv run aide-demo --output-dir outputs/demo
```

## Pull Request Scope

Good contributions for this repository include:

- data collection, selection, and organization improvements
- release-safe configuration and documentation updates
- robustness fixes for public dataset loading and generated processors
- focused tests for release behavior

Please keep downstream fine-tuning, RAG evaluation, plotting, private upload
scripts, and paper figure generation outside this repository unless the project
scope changes.

## Reporting Issues

When filing an issue, include:

- the command you ran
- the config file or relevant config fields
- whether the job was `instruction` or `rag`
- a short excerpt of `outputs/<prompt-hash>/log.txt`
- whether the failure happened during collection, selection, or organization

Do not include API keys, private dataset tokens, or full proprietary data.
