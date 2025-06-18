import unittest
import torch
import os
import sys
import tempfile
import shutil
import yaml
from argparse import Namespace

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Conditional import for entry, assuming it might try to init_distributed early
# For testing, we might want to control this. For now, direct import.
import entry
from model import GPTModel # To check model parameters if needed
from data_builder import DataBuilder # To check vocab size

# Determine if CUDA is available for tests
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

class TestEntryNSPIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        # Ensure sub-directories for checkpoints are also handled if entry script assumes they exist or creates them.
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_test_config(self, nsp_task=False, cpu_fallback=False, use_dummy_dataset=True):
        config_data = {
            'training': {
                'epochs': 1,
                'batch_size': 2,
                'learning_rate': 1e-4, # Small LR for stability in short test
                'warmup_steps': 1,
                'save_every': 10, # Don't save too often
                'eval_every': 5,  # Evaluate a few times
                'log_every': 2,
                'checkpoint_dir': self.checkpoint_dir,
                'nsp_task': nsp_task,
                'nsp_loss_weight': 0.5,
            },
            'data': {
                'seq_len': 32, # Small sequence length
                'max_samples': 20, # Tiny dataset
                'max_eval_tokens': 20 * 32 * 2, # Enough for a few batches
                 # Use fallback (very small, no download) or a tiny specified dataset
                'dataset_name': 'fallback' if use_dummy_dataset else 'wikitext',
                'dataset_config': 'wikitext-2-raw-v1' if not use_dummy_dataset else 'en',
            },
            'model': {
                'dim': 64,       # Small dimension
                'n_layers': 1,   # Minimal layers
                'n_heads': 2,    # Minimal heads
                'max_seq_len': 32, # Consistent with data seq_len
                # vocab_size will be set by DataBuilder
            },
            'hardware': {
                'device': 'cuda' if CUDA_AVAILABLE and not cpu_fallback else 'cpu',
                'cpu_test_attention': cpu_fallback,
            },
            'evaluation': {
                'max_eval_batches': 2, # Quick eval
            },
            'logging': {
                'show_gpu_info': False,
                'test_attention_modes': False, # Disable this for faster integration tests
                'save_training_plots': False,
                'test_generation': False,
            },
            'random_seed': 123,
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        return self.config_path

    def _run_entry_point(self, config_path):
        # Simulate command line arguments
        args = Namespace(
            config=config_path,
            nproc_per_node=1, # Run single process for these tests
            # Set other args to None if they are meant to be taken from config or have defaults in entry.py
            precision=None,
            batch_size=None,
            seq_len=None,
            epochs=None,
            learning_rate=None,
            # NSP args will be in config file for these tests
            nsp_task=None, # Let config file control this
            nsp_loss_weight=None,
            cpu_test_attention=None,
        )

        # Capture stdout/stderr to check logs if necessary (optional)
        # For now, just check for exceptions
        entry.start_actual_training(args)


    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available, skipping GPU NSP test.")
    def test_nsp_training_gpu(self):
        print("\nRunning test_nsp_training_gpu")
        config_file = self._create_test_config(nsp_task=True, cpu_fallback=False, use_dummy_dataset=True)
        try:
            self._run_entry_point(config_file)
            # Check if some checkpoints or metrics were created (basic check)
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
            self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, 'training_metrics.pt')))
        except Exception as e:
            self.fail(f"NSP training on GPU failed with exception: {e}\n{traceback.format_exc()}")


    def test_nsp_training_cpu_fallback(self):
        print("\nRunning test_nsp_training_cpu_fallback")
        config_file = self._create_test_config(nsp_task=True, cpu_fallback=True, use_dummy_dataset=True)

        # To verify CPU fallback mode is active, we'd ideally check logs.
        # For now, we'll rely on successful execution.
        # If we had access to the Trainer instance, we could check trainer.config.device.

        import io
        from contextlib import redirect_stdout

        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                self._run_entry_point(config_file)

            log_content = captured_output.getvalue()
            self.assertTrue("CPU Attention Fallback Mode ENABLED" in log_content)
            self.assertTrue("Model and training will run on CPU" in log_content)

            # Basic check for successful run
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
            self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, 'training_metrics.pt')))

        except Exception as e:
            self.fail(f"NSP training on CPU with fallback failed with exception: {e}\n{traceback.format_exc()}")
        finally:
            print("Captured output for CPU fallback test:\n", captured_output.getvalue())


    def test_standard_training_cpu(self): # Non-NSP, just to ensure CPU path works
        print("\nRunning test_standard_training_cpu (no NSP, CPU attention fallback)")
        config_file = self._create_test_config(nsp_task=False, cpu_fallback=True, use_dummy_dataset=True)
        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                 self._run_entry_point(config_file)
            log_content = captured_output.getvalue()
            self.assertTrue("CPU Attention Fallback Mode ENABLED" in log_content)
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
        except Exception as e:
            self.fail(f"Standard training on CPU with fallback failed: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    # Need to import traceback for the try-except blocks if it's not already global
    import traceback
    unittest.main()
