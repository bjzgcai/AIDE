# Release Checklist

Use this checklist before publishing a new release.

## Repository Hygiene

- Build the public repository from a clean release directory. Do not publish a
  working directory that contains local experiments, paper drafts, credentials,
  caches, or generated outputs.
- No private paths, API keys, private endpoints, or project-specific script names are
  present in the release surface.
- README, configs, and CLI output use the paper-aligned stage names: Data
  Collection, Data Selection, and Data Organization.

## Verification

```bash
uv lock --check
uv run --extra dev ruff check .
uv run python -m unittest discover -s tests
uv run aide-demo --output-dir outputs/demo
uv build
```

## Package Contents

Check source distribution and wheel contents:

```bash
tar -tzf dist/aide_curator-0.1.0.tar.gz
python -m zipfile -l dist/aide_curator-0.1.0-py3-none-any.whl
```

The wheel should contain only the importable `aide` package and package
metadata. The source distribution may include README, LICENSE, configs, docs,
and tests.
