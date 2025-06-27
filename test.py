import argparse
import sys
from pathlib import Path

# Add the project root to sys.path to allow direct import of entry
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from entry import start_actual_training, load_config, merge_config_with_args
except ImportError as e:
    print(f"Error importing from entry.py: {e}")
    print(f"Ensure KernelDev is in your PYTHONPATH or sys.path, or run test.py from the KernelDev directory.")
    sys.exit(1)

def main():
    print("=== Running End-to-End Quick Test Script ===")

    # --- Base Configuration ---
    # Use a fast config as a base, or the default config.yaml
    # Let's use config_fast.yaml if it exists, otherwise config.yaml
    base_config_file = 'config_fast.yaml'
    if not (project_root / base_config_file).exists():
        print(f"Base config {base_config_file} not found, falling back to config.yaml")
        base_config_file = 'config.yaml'
        if not (project_root / base_config_file).exists():
            print(f"ERROR: Default config file {base_config_file} not found. Exiting.")
            sys.exit(1)

    print(f"Using base configuration: {base_config_file}")

    # --- Argument Overrides for Quick Testing ---
    # These will override settings from the base_config_file
    # and any command-line arguments normally parsed by entry.py's ArgumentParser
    # We simulate the args namespace that entry.py expects.

    args_override = argparse.Namespace(
        # Config file itself
        config=base_config_file,

        # Key overrides for a quick test run
        max_samples=1000,        # User request: 1k records
        epochs=1,                # Run for only 1 epoch
        batch_size=8,            # Small batch size for speed
        seq_len=128,             # Shorter sequence length

        # Logging and evaluation frequency
        log_every=10,
        eval_every=50,           # Evaluate a few times if possible
        max_eval_batches=5,      # Limit batches during evaluation (this is an eval_cfg param in entry.py)

        # NSP task (defaults to True in code now, but can be explicit)
        nsp_task=True,           # Explicitly test with NSP enabled
        nsp_loss_weight=0.5,

        # Hardware/Execution
        cpu_test_attention=False, # Use GPU / Triton by default for this test
        precision='bf16',        # Or '16' or '32' depending on test environment focus
                                 # 'bf16' is good for H100 as per user logs

        # Default other CLI args from entry.py to None if they are not being overridden,
        # so they don't interfere with config loading if they were meant to be optional.
        # entry.py's ArgumentParser defines defaults for these if not provided.
        # Our goal here is to primarily override specific values for testing.
        # The merge_config_with_args in entry.py will handle None values appropriately.
        learning_rate=None,      # None means use value from YAML or entry.py default
        nproc_per_node=1         # Forcing single process for this test script
    )

    # Note: max_eval_batches is part of 'evaluation' config in entry.py, not a direct CLI arg.
    # We'll need to handle this by modifying the config dict if we want to set it this way,
    # or rely on it being set in the chosen base_config_file.
    # For this script, we assume base_config_file (e.g. config_fast.yaml) has reasonable eval settings.
    # If direct control is needed, we'd load config, modify, then pass modified config dict to start_actual_training,
    # or add more specific args to the Namespace if entry.py's merge_config_with_args handles them.

    print(f"Applying overrides: max_samples={args_override.max_samples}, epochs={args_override.epochs}, batch_size={args_override.batch_size}, seq_len={args_override.seq_len}")
    print(f"NSP Task: {args_override.nsp_task}, Precision: {args_override.precision}")

    # --- Run the training ---
    # entry.start_actual_training expects the direct output of parse_args()
    # We are providing a namespace that should be compatible.
    try:
        start_actual_training(args_override)
        print("=== Quick Test Script Completed Successfully ===")
    except Exception as e:
        print(f"ERROR: Quick Test Script Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```
