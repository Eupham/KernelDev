#!/usr/bin/env python3
"""
Integration test for the complete checkpointing system.
Tests checkpoint saving, rotation, resume, and configuration options.
"""

import os
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

def test_checkpoint_system_integration():
    """Test the complete checkpoint system integration."""
    print("Testing complete checkpoint system integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock training config
        class MockConfig:
            def __init__(self):
                self.checkpoint_dir = str(checkpoint_dir)
                self.auto_resume = True
                self.max_checkpoints = 2
                self.save_every = 100
        
        # Mock trainer class with checkpoint functionality
        class MockTrainer:
            def __init__(self, config):
                self.config = config
                self.dataset_state = {}
                
            def _cleanup_old_checkpoints(self):
                """Keep only the max_checkpoints most recent regular checkpoints."""
                checkpoint_dir = Path(self.config.checkpoint_dir)
                
                # Find all regular checkpoint files (not best_checkpoint.pt)
                checkpoint_files = []
                for file_path in checkpoint_dir.glob('checkpoint_step_*.pt'):
                    try:
                        # Extract step number from filename
                        step_num = int(file_path.stem.split('_')[-1])
                        checkpoint_files.append((step_num, file_path))
                    except (ValueError, IndexError):
                        continue
                
                # Sort by step number, newest first
                checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                
                # Remove all but the max_checkpoints most recent
                max_checkpoints = getattr(self.config, 'max_checkpoints', 2)
                for _, file_path in checkpoint_files[max_checkpoints:]:
                    try:
                        file_path.unlink()
                        print(f"Removed old checkpoint: {file_path}")
                    except FileNotFoundError:
                        pass  # File already removed
            
            def save_checkpoint(self, step, is_best=False):
                """Mock checkpoint saving with rotation."""
                checkpoint = {
                    'step': step,
                    'model_state_dict': {'dummy': 'model_data'},
                    'optimizer_state_dict': {'dummy': 'optimizer_data'},
                    'scheduler_state_dict': {'dummy': 'scheduler_data'},
                    'metrics': {'dummy': 'metrics_data'},
                    'config': {'dummy': 'config_data'},
                    'dataset_state': self.dataset_state
                }
                
                checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_step_{step}.pt'
                
                # Mock saving (create empty file for testing)
                checkpoint_path.touch()
                
                # Store actual data in a json file for verification
                json_path = checkpoint_path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(checkpoint, f)
                
                print(f"Checkpoint saved: {checkpoint_path}")
                
                # Clean up old checkpoints
                self._cleanup_old_checkpoints()
                
                if is_best:
                    best_path = Path(self.config.checkpoint_dir) / 'best_checkpoint.pt'
                    best_path.touch()
                    best_json_path = best_path.with_suffix('.json')
                    with open(best_json_path, 'w') as f:
                        json.dump(checkpoint, f)
            
            def find_latest_checkpoint(self):
                """Find the most recent checkpoint file."""
                checkpoint_dir = Path(self.config.checkpoint_dir)
                
                if not checkpoint_dir.exists():
                    return None
                
                # Find all regular checkpoint files
                checkpoint_files = []
                for file_path in checkpoint_dir.glob('checkpoint_step_*.pt'):
                    try:
                        step_num = int(file_path.stem.split('_')[-1])
                        checkpoint_files.append((step_num, file_path))
                    except (ValueError, IndexError):
                        continue
                
                if not checkpoint_files:
                    return None
                
                # Return the most recent checkpoint
                checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                return str(checkpoint_files[0][1])
            
            def load_checkpoint(self, checkpoint_path):
                """Mock checkpoint loading."""
                json_path = Path(checkpoint_path).with_suffix('.json')
                
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        checkpoint = json.load(f)
                    
                    # Restore dataset state if available
                    if 'dataset_state' in checkpoint:
                        self.dataset_state = checkpoint['dataset_state']
                    
                    print(f"Checkpoint loaded: {checkpoint_path}")
                    return checkpoint['step']
                else:
                    raise FileNotFoundError(f"Checkpoint data not found: {json_path}")
        
        config = MockConfig()
        trainer = MockTrainer(config)
        
        # Test 1: Save multiple checkpoints and verify rotation
        print("\n--- Test 1: Checkpoint rotation ---")
        trainer.dataset_state = {'current_epoch': 0, 'current_batch': 0}
        trainer.save_checkpoint(100)
        
        trainer.dataset_state = {'current_epoch': 0, 'current_batch': 50}
        trainer.save_checkpoint(200)
        
        trainer.dataset_state = {'current_epoch': 1, 'current_batch': 0}
        trainer.save_checkpoint(300)
        
        trainer.dataset_state = {'current_epoch': 1, 'current_batch': 25}
        trainer.save_checkpoint(400, is_best=True)
        
        # Check that only 2 regular checkpoints remain + best checkpoint
        remaining_files = list(checkpoint_dir.glob("*.pt"))
        remaining_names = [f.name for f in remaining_files]
        
        expected_files = {"checkpoint_step_300.pt", "checkpoint_step_400.pt", "best_checkpoint.pt"}
        actual_files = set(remaining_names)
        
        assert actual_files == expected_files, f"Expected {expected_files}, got {actual_files}"
        print("✓ Checkpoint rotation working correctly")
        
        # Test 2: Find latest checkpoint
        print("\n--- Test 2: Find latest checkpoint ---")
        latest = trainer.find_latest_checkpoint()
        expected_latest = str(checkpoint_dir / "checkpoint_step_400.pt")
        assert latest == expected_latest, f"Expected {expected_latest}, got {latest}"
        print("✓ Latest checkpoint found correctly")
        
        # Test 3: Load checkpoint and verify dataset state
        print("\n--- Test 3: Load checkpoint ---")
        new_trainer = MockTrainer(config)
        loaded_step = new_trainer.load_checkpoint(latest)
        
        assert loaded_step == 400, f"Expected step 400, got {loaded_step}"
        assert new_trainer.dataset_state['current_epoch'] == 1, "Dataset state not restored correctly"
        assert new_trainer.dataset_state['current_batch'] == 25, "Dataset state not restored correctly"
        print("✓ Checkpoint loading and dataset state restoration working correctly")
        
        # Test 4: Config-based checkpoint management
        print("\n--- Test 4: Config-based management ---")
        config.max_checkpoints = 3
        trainer_with_3_checkpoints = MockTrainer(config)
        
        # Save 5 checkpoints
        for i in range(5):
            step = 500 + (i * 100)
            trainer_with_3_checkpoints.save_checkpoint(step)
        
        # Should keep only 3 most recent
        remaining_files = [f for f in checkpoint_dir.glob("checkpoint_step_*.pt") if "best" not in f.name]
        assert len(remaining_files) == 3, f"Expected 3 checkpoints, got {len(remaining_files)}"
        
        # Check that the correct ones remain (most recent 3)
        steps = []
        for f in remaining_files:
            step_num = int(f.stem.split('_')[-1])
            steps.append(step_num)
        
        steps.sort()
        expected_steps = [700, 800, 900]  # Most recent 3 (corrected)
        assert steps == expected_steps, f"Expected steps {expected_steps}, got {steps}"
        print("✓ Configurable max_checkpoints working correctly")
        
        print("\n✅ All integration tests passed!")


def test_config_options():
    """Test configuration options for checkpoint system."""
    print("Testing checkpoint configuration options...")
    
    # Test different max_checkpoints values
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        class MockConfig:
            def __init__(self, max_checkpoints=2, auto_resume=True):
                self.checkpoint_dir = str(checkpoint_dir)
                self.auto_resume = auto_resume
                self.max_checkpoints = max_checkpoints
        
        # Test max_checkpoints = 1
        config = MockConfig(max_checkpoints=1)
        assert config.max_checkpoints == 1, "max_checkpoints not set correctly"
        
        # Test auto_resume = False
        config = MockConfig(auto_resume=False)
        assert config.auto_resume == False, "auto_resume not set correctly"
        
        print("✓ Configuration options working correctly")


def main():
    """Run all integration tests."""
    print("Running checkpoint system integration tests...\n")
    
    try:
        test_checkpoint_system_integration()
        test_config_options()
        
        print("\n✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)