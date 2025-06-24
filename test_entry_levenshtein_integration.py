import unittest
import torch
import os
import sys
import tempfile
import shutil
import yaml
from argparse import Namespace
import io
from contextlib import redirect_stdout
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import entry
from model import GPTModel
from data_builder import DataBuilder

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

class TestEntryLevenshteinIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_test_config(self, use_lev_task=True, cpu_fallback=False, use_dummy_dataset=True):
        config_data = {
            'training': {
                'epochs': 1,
                'batch_size': 2,
                'learning_rate': 1e-4,
                'warmup_steps': 1,
                'save_every': 10,
                'eval_every': 5,
                'log_every': 2,
                'checkpoint_dir': self.checkpoint_dir,
                'use_levenshtein_task': use_lev_task, # Updated
                'levenshtein_loss_weight': 0.1,    # Updated
            },
            'data': {
                'seq_len': 32,
                'max_samples': 20,
                'max_eval_tokens': 20 * 32 * 2,
                'dataset_name': 'hf-internal-testing/tiny_c4_ संस्कृत' if use_dummy_dataset else 'wikitext',
                'dataset_config': 'sa' if use_dummy_dataset else 'wikitext-2-raw-v1',
            },
            'model': {
                'dim': 64,
                'n_layers': 1,
                'n_heads': 2,
                'max_seq_len': 32,
                'use_cls_prefix_attention': True, # For Levenshtein CLS token
                'lm_self_critique_base_penalty': 0.3, # New
                'lm_self_critique_reward_max': 0.3,   # New
                # vocab_size will be set by DataBuilder
            },
            'hardware': {
                'device': 'cuda' if CUDA_AVAILABLE and not cpu_fallback else 'cpu',
                'cpu_test_attention': cpu_fallback,
            },
            'evaluation': {'max_eval_batches': 2},
            'logging': {
                'show_gpu_info': False, 'test_attention_modes': False,
                'save_training_plots': False, 'test_generation': False,
            },
            'random_seed': 123,
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        return self.config_path

    def _run_entry_point(self, config_path, cli_overrides=None):
        args_dict = {
            'config': config_path,
            'nproc_per_node': 1,
            'precision': None, 'batch_size': None, 'seq_len': None, 'epochs': None, 'learning_rate': None,
            'use_levenshtein_task': None, 'levenshtein_loss_weight': None,
            'lm_self_critique_base_penalty': None, 'lm_self_critique_reward_max': None,
            'cpu_test_attention': None,
            'use_cls_prefix_attention': None,
        }
        if cli_overrides:
            args_dict.update(cli_overrides)

        # Handle boolean flags that are action='store_true'
        if args_dict.get('cpu_test_attention') is False: # argparse sets False if not present for store_true
             del args_dict['cpu_test_attention'] # Remove if False, presence means True

        args = Namespace(**args_dict)
        entry.start_actual_training(args)

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available, skipping GPU Levenshtein test.")
    def test_levenshtein_training_gpu(self):
        print("\nRunning test_levenshtein_training_gpu")
        config_file = self._create_test_config(use_lev_task=True, cpu_fallback=False)
        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                self._run_entry_point(config_file)

            log_content = captured_output.getvalue()
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
            self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, 'training_metrics.pt')))
            self.assertIn("Levenshtein Batch shapes", log_content) # Check for Levenshtein data log
            self.assertIn("LM Comp:", log_content) # Check for Levenshtein training log
            self.assertIn("Lev Aux (shuf):", log_content)
        except Exception as e:
            self.fail(f"Levenshtein training on GPU failed: {e}\n{traceback.format_exc()}\nLogs:\n{captured_output.getvalue()}")

    def test_levenshtein_training_cpu_fallback(self):
        print("\nRunning test_levenshtein_training_cpu_fallback")
        config_file = self._create_test_config(use_lev_task=True, cpu_fallback=True)
        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                self._run_entry_point(config_file)

            log_content = captured_output.getvalue()
            self.assertIn("CPU Attention Fallback Mode ENABLED", log_content)
            # self.assertIn("Model and training will run on CPU", log_content) # This log was removed/changed in entry.py
            self.assertIn("Effective device determined as: cpu", log_content)


            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
            self.assertIn("Levenshtein Batch shapes", log_content)
            self.assertIn("LM Comp:", log_content)
        except Exception as e:
            self.fail(f"Levenshtein training on CPU fallback failed: {e}\n{traceback.format_exc()}\nLogs:\n{captured_output.getvalue()}")

    def test_pure_lm_training_cpu_fallback(self):
        print("\nRunning test_pure_lm_training_cpu_fallback (no Levenshtein)")
        config_file = self._create_test_config(use_lev_task=False, cpu_fallback=True)
        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                 self._run_entry_point(config_file)
            log_content = captured_output.getvalue()
            self.assertIn("CPU Attention Fallback Mode ENABLED", log_content)
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
            self.assertNotIn("Levenshtein Batch shapes", log_content) # Ensure Levenshtein specific logs are not present
            self.assertNotIn("Lev Aux (shuf):", log_content)
        except Exception as e:
            self.fail(f"Pure LM training on CPU fallback failed: {e}\n{traceback.format_exc()}\nLogs:\n{captured_output.getvalue()}")

    def test_levenshtein_training_cls_no_prefix_attention(self):
        print("\nRunning test_levenshtein_training_cls_no_prefix_attention")
        use_cpu_for_this_test = not CUDA_AVAILABLE
        config_file = self._create_test_config(use_lev_task=True, cpu_fallback=use_cpu_for_this_test)

        temp_config = entry.load_config(config_file)
        temp_config.setdefault('model', {})['use_cls_prefix_attention'] = False
        with open(config_file, 'w') as f: yaml.dump(temp_config, f)

        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                self._run_entry_point(config_file)

            log_content = captured_output.getvalue()
            self.assertTrue(any(fname.startswith('checkpoint_step_') for fname in os.listdir(self.checkpoint_dir)))
            # Check logs for model init print. The exact string changed in model.py before.
            # Let's check for the relevant part.
            self.assertIn("use_cls_prefix_attention=False", log_content, "Model init log for use_cls_prefix_attention=False not found.")

        except Exception as e:
            self.fail(f"Levenshtein training with CLS no prefix attention failed: {e}\n{traceback.format_exc()}\nLogs:\n{captured_output.getvalue()}")

    def test_generation_with_cls_token_and_prefix_flags(self):
        print("\nRunning test_generation_with_cls_token_and_prefix_flags")
        cls_token_id_for_test = 0
        model_config_dict = {
            'vocab_size': 257, # Adjusted for CLS if cls_token_id_for_test is e.g. 256
            'dim': 32, 'n_layers': 1, 'n_heads': 2, 'max_seq_len': 64,
            'cls_token_id': cls_token_id_for_test, # Relevant for Levenshtein head and prefix attention
            'use_cls_prefix_attention': True # Model IS capable of prefix attention
        }
        # Ensure cls_token_id is less than vocab_size if it's not managed by DataBuilder here
        if cls_token_id_for_test >= model_config_dict['vocab_size']:
            model_config_dict['vocab_size'] = cls_token_id_for_test + 1

        model = GPTModel(**model_config_dict)
        model.eval()
        model.to(DEVICE)

        prompt_ids = torch.tensor([[cls_token_id_for_test, 10, 20, 30]], device=DEVICE)

        try:
            print("  Generating with use_prefix_attention_in_prompt=False...")
            generated_ids_prefix_off = model.generate(
                prompt_ids.clone(), max_new_tokens=5, use_prefix_attention_in_prompt=False
            )
            self.assertEqual(generated_ids_prefix_off.shape, (1, prompt_ids.shape[1] + 5))

            print("  Generating with use_prefix_attention_in_prompt=True...")
            generated_ids_prefix_on = model.generate(
                prompt_ids.clone(), max_new_tokens=5, use_prefix_attention_in_prompt=True
            )
            self.assertEqual(generated_ids_prefix_on.shape, (1, prompt_ids.shape[1] + 5))

            if not torch.equal(generated_ids_prefix_off, generated_ids_prefix_on):
                print("  Generated outputs differ with prefix flag, exercising different paths.")
            else:
                print("  Warning: Generated outputs are identical. This might be okay for tiny models/short generation.")
        except Exception as e:
            self.fail(f"Generation test with CLS prefix flags failed: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    unittest.main()
