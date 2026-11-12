# Python Style Guide

This project follows [PEP 8](https://peps.python.org/pep-0008/). This guide lists only the decisions that go beyond or deviate from PEP 8.

## General

- Target Python **3.12**.
- Write all identifiers, comments and docstrings in **English**.
- Formatting and linting are enforced by [Ruff](https://docs.astral.sh/ruff/).

## Formatting

- Maximum line length: **88** characters.
- Use **double quotes** for strings.
- Indent with 4 spaces, never tabs.
- Use trailing commas in multi-line collections and calls.

## Naming

| Element             | Style                 | Example            |
| ------------------- | --------------------- | ------------------ |
| Module / package    | `snake_case`          | `servo_control.py` |
| Class / exception   | `PascalCase`          | `ServoController`  |
| Function / method   | `snake_case`          | `move_to()`        |
| Variable / argument | `snake_case`          | `target_angle`     |
| Constant            | `UPPER_SNAKE_CASE`    | `MAX_ANGLE`        |
| Internal / private  | `_leading_underscore` | `_send_command()`  |

- Exception names end with `Error`.
- Include units in names where ambiguous: `timeout_s`, `angle_deg`.

## Imports

- One import per line; absolute imports only.
- Group in order: standard library, third-party, local. Ruff sorts them.
- No wildcard imports (`from x import *`).

## Type Hints

- Annotate **all** function and method parameters and return values,
  including private ones and tests.
- Annotate module-level constants and class attributes.
- Use built-in generics and `|` unions: `list[int]`, `str | None`.

## Docstrings

- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Every public module, class, function and method has a docstring.
- Do not repeat types in docstrings; they are in the signature.

## Functions

- A function body has at most **10 lines**, excluding signature and docstring.
- Ruff enforces this as a limit of 10 statements; the line limit is checked in review.
- Cyclomatic complexity is at most **5** per function.

## Comments

- Explain *why*, not *what*.
- Keep comments short and up to date; delete obsolete ones.

## Tooling

Ruff is configured in [pyproject.toml](../pyproject.toml). Run before committing:

```sh
ruff format .
ruff check --fix .
```
