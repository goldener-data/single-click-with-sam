# Single click with SAM - Does SAM really segment anything from a single click?

Experiment qualifying the performances of the Segment Anything model in the situation of single click annotation on data not used during training.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/goldener-data/single-click-with-sam.git
cd single-click-with-sam
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

## Run an experiment on a dataset

The experiment are managed by leveraging Hydra, Pixeltable and Mlflow.

To run the experiment with the default settings, you can use the following command:
```bash
uv run python -m single_click_with_sam
```

In order to run the experiment with different settings (dataset, number of data, number of random points, ...),
see the Hydra documentation to update the configuration file or to use the command line options.

The images and the segmentation masks issued by SAM can be analysed using:

```bash
uv run python -m sam_prediction_visualization
```

## Contributing

We welcome contributions to `single-click-with-sam` Here's how you can help:

### Getting Started

1. Fork the repository
2. Clone your fork
3. Install the dependencies
4. Create your branch and make your proposals
5. Push to your fork and create a pull request
6. The PR will be automatically tested by GitHub Actions
7. A maintainer will review your PR and may request changes
8. Once approved, your PR will be merged

### Development

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
