# Contributing

## Development setup

Install Python 3.12 or newer, then create and activate a virtual environment and install Ruff:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install ruff
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Before submitting changes, run:

```sh
ruff format .
ruff check --fix .
```

See the [Python style guide](docs/STYLEGUIDE.md) for project conventions.
