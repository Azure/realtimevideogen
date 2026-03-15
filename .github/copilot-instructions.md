See AGENTS.md for full repository instructions. Key rules are summarised below.

## Project

Real-Time Multi-Modal Generation — modular, adaptive serving stack for video/audio/image generation on a Kubernetes cluster with GPU nodes.

## Quick reference

- **Python 3.12**; all functions must have type annotations (`mypy` strict).
- **Line length**: 120 characters max (flake8).
- **Async first**: use `async`/`await` and `aiofiles` for I/O; Quart as the web framework.
- **Logging**: use the logger from `console_utils.py`; no f-strings in log calls.
- **K8s namespace**: always `rtgen`.
- **No secrets in source**: use environment variables or Azure Key Vault.

## Lint & test (run before every PR)

```bash
flake8 . tests
mypy --install-types --non-interactive --ignore-missing-imports .
pytest --ignore=tests/simulator --ignore=tests/streamwise --ignore=tests/streamwise_app -vv
```

## Must / Never

- **Must** add or update tests for every functional change.
- **Must** keep `services.json` valid (`python -m json.tool services.json`).
- **Never** commit credentials, tokens, or secrets.
- **Never** remove or weaken existing tests.
