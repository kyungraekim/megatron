# Test Guide

> Parent: [../CLAUDE.md](../CLAUDE.md)

## Structure

- `unit_tests/` — Fast, isolated component tests
- `functional_tests/` — End-to-end multi-GPU integration tests
- `test_utils/` — Shared utilities and YAML test recipes (`test_utils/recipes/`)

## Markers

- `@pytest.mark.internal` — Internal-only tests
- `@pytest.mark.flaky` — Timing-sensitive tests that may intermittently fail
- `@pytest.mark.flaky_in_dev` — Flaky only in development environment (stable in CI)

## Gotchas

- Distributed tests **must** call `destroy_model_parallel()` in teardown. The `conftest.py` fixtures automate this — use them.
- Use `tmp_path_dist_ckpt` fixture for checkpoint tests. It handles cleanup across all ranks automatically.
- YAML recipes in `test_utils/recipes/` define test configuration templates. Check these before creating new test configs.
- Some tests (optimizer, checkpoint) exceed 700 lines. Check existing patterns before refactoring.

## Commands

```bash
pytest tests/unit_tests/                                    # All unit tests
pytest tests/unit_tests/transformer/test_attention.py       # Specific test
pytest --cov=megatron.core tests/unit_tests/                # With coverage
pytest -m "not flaky" tests/                                # Skip flaky
pytest tests/functional_tests/                              # Integration tests
```
