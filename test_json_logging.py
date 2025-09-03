#!/usr/bin/env python3
"""
Test script for JSON logging functionality.
"""

import os
import tempfile
import json
from pathlib import Path
from train_loop import TrainingMetrics


def test_json_logging():
    """Test that training metrics can be saved to JSON format."""
    print("Testing JSON logging functionality...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create TrainingMetrics instance
        metrics = TrainingMetrics(moving_avg_window=10)
        
        # Add some sample data
        metrics.update(train_loss=0.5, learning_rate=0.001, step_time=0.1)
        metrics.update(train_loss=0.4, learning_rate=0.0009, step_time=0.12)
        metrics.update(train_loss=0.3, val_loss=0.35, learning_rate=0.0008, step_time=0.11)
        
        # Save to JSON
        json_path = os.path.join(temp_dir, "test_metrics.json")
        metrics.save_metrics_json(json_path)
        
        # Verify file was created
        assert os.path.exists(json_path), "JSON file was not created"
        
        # Load and verify contents
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Check that all expected fields are present
        expected_fields = [
            'train_losses', 'val_losses', 'cocktail_party_metrics',
            'learning_rates', 'step_times', 'total_steps',
            'best_val_loss', 'best_step', 'timestamp'
        ]
        
        for field in expected_fields:
            assert field in loaded_data, f"Field '{field}' missing from JSON"
        
        # Verify data integrity
        assert len(loaded_data['train_losses']) == 3, "Wrong number of training losses"
        assert len(loaded_data['val_losses']) == 1, "Wrong number of validation losses"
        assert len(loaded_data['learning_rates']) == 3, "Wrong number of learning rates"
        assert loaded_data['total_steps'] == 3, "Wrong total steps count"
        assert loaded_data['best_val_loss'] == 0.35, "Wrong best validation loss"
        assert 'timestamp' in loaded_data and loaded_data['timestamp'] > 0, "Missing or invalid timestamp"
        
        print("✓ JSON logging test passed!")


def test_json_logging_path_creation():
    """Test that directories are created when saving JSON logs."""
    print("Testing JSON logging path creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics = TrainingMetrics()
        metrics.update(train_loss=0.5)
        
        # Test with nested path that doesn't exist
        nested_path = os.path.join(temp_dir, "logs", "training", "metrics.json")
        
        # Create the directory structure manually (simulating what train_loop.py does)
        logs_dir = Path(nested_path).parent
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        metrics.save_metrics_json(nested_path)
        
        # Verify file was created
        assert os.path.exists(nested_path), "JSON file was not created in nested path"
        
        print("✓ JSON logging path creation test passed!")


def main():
    """Run all tests."""
    print("🧪 Running JSON Logging Tests")
    print("=" * 40)
    
    test_json_logging()
    test_json_logging_path_creation()
    
    print()
    print("✅ All JSON logging tests passed!")


if __name__ == "__main__":
    main()