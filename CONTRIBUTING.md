# Contributing

```bash
git clone https://github.com/sabeeh-saad/placecell && cd placecell
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
ruff check .
ruff format --check .
mypy
pytest --cov
```

Ruff also formats Python examples in Markdown. Keep the Ruff dependency and
`required-version` in `pyproject.toml` aligned with `.pre-commit-config.yaml` so
local checks, commit hooks, and CI use the same version.

Rules that keep the code base honest:

- Every public function is typed and passes `mypy --strict`. Every module has a docstring that says why it exists.
- Behaviour is tested, not just called. A pull request that changes behaviour comes with the test that would have failed before it.
- No network in tests. Providers and transports are injected; the fakes in `tests/conftest.py` show how.
- One embedding model per collection, evidence never inside the store, filters pushed down into backends. Changes that bend these need a design note in the pull request.
- Commit messages say what changed and why, in one line if possible.
