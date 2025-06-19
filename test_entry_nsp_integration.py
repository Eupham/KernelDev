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

    def _run_entry_point(self, config_path, cli_overrides=None):
        # Simulate command line arguments
        # Start with defaults that won't override config unless specified in cli_overrides
        args_dict = {
            'config': config_path,
            'nproc_per_node': 1, # Run single process for these tests
            'precision': None,
            'batch_size': None,
            'seq_len': None,
            'epochs': None,
            'learning_rate': None,
            'nsp_task': None,
            'nsp_loss_weight': None,
            'cpu_test_attention': None, # This is action='store_true', so presence matters if not None
            'use_cls_prefix_attention': None,
        }
        if cli_overrides:
            args_dict.update(cli_overrides)

        args = Namespace(**args_dict)

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
                 self._run_entry_point(config_file) # No specific CLI overrides for this one beyond what config does
            log_content = captured_output.getvalue()
            self.assertTrue("CPU Attention Fallback Mode ENABLED" in log_content)
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
        except Exception as e:
            self.fail(f"Standard training on CPU with fallback failed: {e}\n{traceback.format_exc()}")

    def test_nsp_training_cls_no_prefix_attention(self):
        print("\nRunning test_nsp_training_cls_no_prefix_attention")
        # This test runs with NSP on, but CLS prefix attention specifically disabled in model config.
        # The cpu_fallback here is to ensure it runs on CPU if CUDA isn't available,
        # or to test this config with CPU fallback.
        use_cpu_for_this_test = not CUDA_AVAILABLE
        config_file = self._create_test_config(
            nsp_task=True,
            cpu_fallback=use_cpu_for_this_test
            # model: use_cls_prefix_attention will be set to false directly in the dict below
        )

        # Modify the created config file to set use_cls_prefix_attention to false
        temp_config = entry.load_config(config_file)
        temp_config.setdefault('model', {})['use_cls_prefix_attention'] = False
        with open(config_file, 'w') as f:
            yaml.dump(temp_config, f)

        print(f"Modified config for test_nsp_training_cls_no_prefix_attention: {temp_config['model']}")

        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                self._run_entry_point(config_file) # CLI override for cpu_test_attention is via config

            # Check for successful completion
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
            self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, 'training_metrics.pt')))

            # Check logs for model init print
            log_content = captured_output.getvalue()
            expected_init_log = "use_cls_prefix_attention=False"
            self.assertTrue(expected_init_log in log_content, f"Expected log '{expected_init_log}' not found.")

        except Exception as e:
            self.fail(f"NSP training with CLS no prefix attention failed: {e}\n{traceback.format_exc()}")
        finally:
            print("Captured output for NSP no prefix attention test:\n", captured_output.getvalue())

    def test_generation_with_cls_token_and_prefix_flags(self):
        print("\nRunning test_generation_with_cls_token_and_prefix_flags")
        cls_token_id_for_test = 0 # Using 0 as CLS for this isolated test.
                                  # Ensure vocab_size is large enough.
        model_config_dict = {
            'vocab_size': 258,
            'dim': 32, 'n_layers': 1, 'n_heads': 2, 'max_seq_len': 64,
            'nsp_task': True,
            'cls_token_id': cls_token_id_for_test,
            'use_cls_prefix_attention': True # Model IS capable of prefix attention
        }
        model = GPTModel(**model_config_dict)
        model.eval()
        model.to(DEVICE) # Move model to appropriate device

        prompt_ids = torch.tensor([[cls_token_id_for_test, 10, 20, 30]], device=DEVICE)

        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Prompt device: {prompt_ids.device}")

        try:
            # Test 1: force_disable_prefix_attention = True (so use_prefix_attention_in_prompt=False)
            print("  Generating with use_prefix_attention_in_prompt=False...")
            generated_ids_prefix_off = model.generate(
                prompt_ids.clone(),
                max_new_tokens=5,
                use_prefix_attention_in_prompt=False
            )
            self.assertEqual(generated_ids_prefix_off.shape, (1, prompt_ids.shape[1] + 5))

            # Test 2: force_disable_prefix_attention = False (so use_prefix_attention_in_prompt=True)
            print("  Generating with use_prefix_attention_in_prompt=True...")
            generated_ids_prefix_on = model.generate(
                prompt_ids.clone(),
                max_new_tokens=5,
                use_prefix_attention_in_prompt=True
            )
            self.assertEqual(generated_ids_prefix_on.shape, (1, prompt_ids.shape[1] + 5))

            # Optional: check if outputs differ. This can be flaky.
            # For this test, we mainly care that both paths execute without error.
            if not torch.equal(generated_ids_prefix_off, generated_ids_prefix_on):
                print("  Generated outputs differ with prefix flag, exercising different paths.")
            else:
                # This could happen if the model is too small/random or if the logic isn't effective.
                print("  Warning: Generated outputs are identical. This might be okay for tiny models/short generation, but double-check prefix attention logic if this is unexpected.")

        except Exception as e:
            self.fail(f"Generation test with CLS prefix flags failed: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    # Need to import traceback for the try-except blocks if it's not already global
    import traceback
    import io # For capturing stdout
    from contextlib import redirect_stdout
    unittest.main()
