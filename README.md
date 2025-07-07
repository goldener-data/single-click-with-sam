# single-click-with-sam

Experiment qualifying the performances of the Segment Anything model in the situation of single clock annotation on data not used during training.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/goldener-data/goldener.git
cd goldener
```

2. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and activate a virtual environment (optional but recommended):
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
```

4. Install the package and its dependencies:
```bash
uv sync
```


## Development

To set up the development environment:

1. Install development dependencies:
```bash
uv sync --all-extras  # Install all dependencies including development dependencies
```

2. Run tests:
```bash
pytest .
```

3. Run type checking with mypy:
```bash
uv run mypy .
```

4. Run linting with ruff:
```bash
# Run all checks
uv run ruff check .

# Format code
uv run ruff format .
```

5. Set up pre-commit hooks:
```bash
# Install git hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

The pre-commit hooks will automatically run:
- mypy for type checking
- ruff for linting and formatting
- pytest for tests

whenever you make a commit.
