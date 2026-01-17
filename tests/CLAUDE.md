# CLAUDE.md - tests

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Test infrastructure: unit tests, functional tests, fixtures, YAML recipes
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

## Overview

This directory contains the test infrastructure for Megatron-LM. It includes 188+ unit
tests for validating components in isolation, functional tests for end-to-end behavior,
shared fixtures, and YAML-based test recipes. Tests use pytest with custom markers for
controlling test execution across different environments.

The purpose is to ensure correctness and prevent regressions across parallelism
strategies, optimizer implementations, checkpoint systems, and model configurations.
Tests follow a clear separation: fast unit tests in `unit_tests/`, integration tests in
`functional_tests/`, and reusable utilities in `test_utils/`.

**Key Concepts**:
- **Unit Tests** (188 files): Isolated, fast tests for individual components
- **Functional Tests** (14 files): End-to-end integration tests across modules
- **Fixtures**: Shared setup/teardown in conftest.py with distributed cleanup
- **Pytest Markers**: @pytest.mark.internal, @pytest.mark.flaky,
  @pytest.mark.flaky_in_dev
- **YAML Recipes**: Test configuration templates in test_utils/recipes/

## File Map

Key test files (>100 lines):

| File | Lines | Purpose |
|------|-------|---------|
| unit_tests/test_optimizer.py | 752 | Distributed optimizer, gradients, schedulers |
| unit_tests/test_argument_utils.py | 643 | Training argument parsing and validation |
| unit_tests/test_hyper_comm_grid.py | 544 | Hyperparameter and communication grid |
| unit_tests/test_parallel_state.py | 501 | Process group management |
| unit_tests/test_utils.py | 459 | General utility functions |
| unit_tests/test_checkpointing.py | 379 | Distributed checkpoint/restore |
| unit_tests/test_tokenizer.py | 278 | Tokenization and sequence handling |
| unit_tests/test_optimizer_param_scheduler.py | 251 | LR and optimizer scheduling |
| unit_tests/conftest.py | 110 | Pytest fixtures and distributed setup |

## Architecture

### Directory Structure

```
tests/
├── unit_tests/           # 188 test files (fast, isolated)
│   ├── test_*.py        # Individual component tests
│   ├── conftest.py      # Shared fixtures (distributed, cleanup)
│   └── ...
├── functional_tests/     # 14 test files (end-to-end)
│   ├── test_cases/      # Integration test cases
│   └── ...
└── test_utils/          # 16 utility files
    ├── recipes/         # YAML test configuration templates
    └── ...
```

### Test Patterns

1. **Isolated Tests**: No distributed setup; test single components directly
2. **Distributed Tests**: Use fixtures to spawn multi-GPU setups via torchrun
3. **Parametrized Tests**: @pytest.mark.parametrize for testing multiple configs
4. **Markers**: @pytest.mark.internal, @pytest.mark.flaky to control execution

### Fixture System

Key fixtures in conftest.py:
- `tmp_path_dist_ckpt`: Temporary directory for distributed checkpoint tests
- `distributed_group_`: Initialize process groups for multi-GPU tests
- Cleanup hooks: Destroy process groups after tests complete

## Common Tasks

### Run all unit tests
```bash
pytest tests/unit_tests/
```

### Run specific test file
```bash
pytest tests/unit_tests/test_optimizer.py
```

### Run with coverage
```bash
pytest --cov=megatron.core tests/unit_tests/
```

### Run internal tests only
```bash
pytest -m internal tests/
```

### Run excluding flaky tests
```bash
pytest -m "not flaky and not flaky_in_dev" tests/unit_tests/
```

### Run functional integration tests
```bash
pytest tests/functional_tests/
```

## Dependencies

- **Depends On**: megatron/core/, megatron/training/, megatron/
- **Test Frameworks**: pytest (with custom markers), torch.testing.assert_close,
  pytest-cov
- **Utilities**: torch, torch.distributed, transformer_engine (optional)
- **Recipes**: YAML config templates in test_utils/recipes/

## Gotchas

1. **Distributed Tests**: Some tests require multi-GPU setup. Tests using
   `distributed.init_process_group()` must call `destroy_process_group()` in
   fixtures. conftest.py handles this automatically.

2. **Flaky Markers**: Use @pytest.mark.flaky for timing-sensitive tests. Use
   @pytest.mark.flaky_in_dev for tests flaky only in development environment.

3. **Temporary Files**: Use tmp_path_dist_ckpt fixture for checkpoint testing;
   it handles cleanup across all ranks automatically.

4. **Large Tests**: Some tests (optimizer, checkpoint) exceed 700 lines. Check
   existing patterns before refactoring to avoid breaking established patterns.

5. **Process Groups**: Always clean up process groups after distributed tests.
   Use conftest.py fixtures to automate this.
