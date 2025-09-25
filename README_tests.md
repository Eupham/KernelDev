# Test Overview

The repository currently ships a lean collection of Python test modules that
focus on checkpointing behaviour and JSON logging utilities.  GPU-specific
attention demos have been retired, so the remaining suites run entirely on the
CPU.

## Available Test Modules

- ``test_checkpointing.py`` – Exercises the standalone checkpoint helpers.
- ``test_checkpoint_integration.py`` – Covers the end-to-end checkpoint flow
  during a mock training session.
- ``test_json_integration.py`` – Validates JSON metric logging and file
  creation.
- ``test_categorization_demo.py`` – Provides a textual walkthrough of how the
  active suites fit together.

## Running the Tests

```bash
# Discover everything (requires optional dependencies such as torch)
python -m pytest

# Or run individual modules with the standard interpreter
python test_checkpointing.py
python test_json_integration.py
```

When optional dependencies are missing, ``pytest`` collection will fail.  Running
the individual modules can still be useful in lightweight environments.
