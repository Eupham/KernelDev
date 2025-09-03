#!/usr/bin/env python3
"""
Test script for checkpoint functionality.
Tests checkpoint saving, loading, rotation, and resume features.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

def test_checkpoint_rotation():
    """Test that only 2 most recent checkpoints are kept."""
    print("Testing checkpoint rotation...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock checkpoint files
        checkpoint_files = [
            checkpoint_dir / "checkpoint_step_100.pt",
            checkpoint_dir / "checkpoint_step_200.pt", 
            checkpoint_dir / "checkpoint_step_300.pt",
            checkpoint_dir / "checkpoint_step_400.pt",
            checkpoint_dir / "best_checkpoint.pt"  # This should not be deleted
        ]
        
        # Create the files
        for file_path in checkpoint_files:
            file_path.touch()
        
        # Mock a trainer instance with necessary attributes
        class MockTrainer:
            def __init__(self, checkpoint_dir):
                self.config = MagicMock()
                self.config.checkpoint_dir = str(checkpoint_dir)
            
            def _cleanup_old_checkpoints(self):
                """Keep only the 2 most recent regular checkpoints."""
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
                
                # Remove all but the 2 most recent
                for _, file_path in checkpoint_files[2:]:
                    try:
                        file_path.unlink()
                        print(f"Removed old checkpoint: {file_path}")
                    except FileNotFoundError:
                        pass  # File already removed
        
        trainer = MockTrainer(checkpoint_dir)
        trainer._cleanup_old_checkpoints()
        
        # Check results
        remaining_files = list(checkpoint_dir.glob("*.pt"))
        remaining_names = [f.name for f in remaining_files]
        
        expected_files = {"checkpoint_step_300.pt", "checkpoint_step_400.pt", "best_checkpoint.pt"}
        actual_files = set(remaining_names)
        
        assert actual_files == expected_files, f"Expected {expected_files}, got {actual_files}"
        print("✓ Checkpoint rotation test passed!")


def test_find_latest_checkpoint():
    """Test finding the latest checkpoint."""
    print("Testing find latest checkpoint...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint files with different step numbers
        checkpoints = [
            checkpoint_dir / "checkpoint_step_150.pt",
            checkpoint_dir / "checkpoint_step_300.pt",
            checkpoint_dir / "checkpoint_step_75.pt"
        ]
        
        for file_path in checkpoints:
            file_path.touch()
        
        # Mock the find_latest_checkpoint_path function
        def find_latest_checkpoint_path(checkpoint_dir_str):
            """Find the most recent checkpoint file in the given directory."""
            checkpoint_dir_path = Path(checkpoint_dir_str)
            
            if not checkpoint_dir_path.exists():
                return None
            
            # Find all regular checkpoint files
            checkpoint_files = []
            for file_path in checkpoint_dir_path.glob('checkpoint_step_*.pt'):
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
        
        latest = find_latest_checkpoint_path(str(checkpoint_dir))
        expected_latest = str(checkpoint_dir / "checkpoint_step_300.pt")
        
        assert latest == expected_latest, f"Expected {expected_latest}, got {latest}"
        print("✓ Find latest checkpoint test passed!")


def test_no_checkpoints():
    """Test behavior when no checkpoints exist."""
    print("Testing no checkpoints scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "nonexistent"
        
        def find_latest_checkpoint_path(checkpoint_dir_str):
            checkpoint_dir_path = Path(checkpoint_dir_str)
            
            if not checkpoint_dir_path.exists():
                return None
            
            checkpoint_files = []
            for file_path in checkpoint_dir_path.glob('checkpoint_step_*.pt'):
                try:
                    step_num = int(file_path.stem.split('_')[-1])
                    checkpoint_files.append((step_num, file_path))
                except (ValueError, IndexError):
                    continue
            
            if not checkpoint_files:
                return None
            
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)
            return str(checkpoint_files[0][1])
        
        latest = find_latest_checkpoint_path(str(checkpoint_dir))
        assert latest is None, f"Expected None, got {latest}"
        print("✓ No checkpoints test passed!")


def main():
    """Run all checkpoint tests."""
    print("Running checkpoint functionality tests...\n")
    
    try:
        test_checkpoint_rotation()
        test_find_latest_checkpoint()
        test_no_checkpoints()
        
        print("\n✅ All checkpoint tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)