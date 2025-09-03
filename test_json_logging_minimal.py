#!/usr/bin/env python3
"""
Minimal test script for JSON logging functionality.
"""

import os
import tempfile
import json
import sys
import time

# Add current directory to Python path
sys.path.insert(0, '/home/runner/work/KernelDev/KernelDev')

# Mock the external dependencies for testing
class MockTorch:
    def save(self, data, path):
        pass

    class device:
        def __init__(self, device_type):
            self.type = device_type

    def cuda_is_available(self):
        return False

sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = type(sys)('mock')
sys.modules['torch.nn.functional'] = type(sys)('mock')
sys.modules['torch.utils'] = type(sys)('mock')
sys.modules['torch.utils.data'] = type(sys)('mock')
sys.modules['torch.utils.data.distributed'] = type(sys)('mock')
sys.modules['torch.distributed'] = type(sys)('mock')
sys.modules['torch.nn.parallel'] = type(sys)('mock')
sys.modules['torch.distributions'] = type(sys)('mock')
sys.modules['numpy'] = type(sys)('mock')
sys.modules['matplotlib'] = type(sys)('mock')
sys.modules['matplotlib.pyplot'] = type(sys)('mock')
sys.modules['data_builder'] = type(sys)('mock')

# Import numpy functionality
import math

# Manual implementation of numpy functions for testing
def mean(arr):
    return sum(arr) / len(arr) if arr else 0.0

def var(arr):
    if len(arr) < 2:
        return 0.0
    m = mean(arr)
    return sum((x - m) ** 2 for x in arr) / len(arr)

# Mock numpy
sys.modules['numpy'].mean = mean
sys.modules['numpy'].var = var

# Now import our module
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
        assert len(loaded_data['train_losses']) == 3, f"Wrong number of training losses: {len(loaded_data['train_losses'])}"
        assert len(loaded_data['val_losses']) == 1, f"Wrong number of validation losses: {len(loaded_data['val_losses'])}"
        assert len(loaded_data['learning_rates']) == 3, f"Wrong number of learning rates: {len(loaded_data['learning_rates'])}"
        assert loaded_data['total_steps'] == 3, f"Wrong total steps count: {loaded_data['total_steps']}"
        assert loaded_data['best_val_loss'] == 0.35, f"Wrong best validation loss: {loaded_data['best_val_loss']}"
        assert 'timestamp' in loaded_data and loaded_data['timestamp'] > 0, "Missing or invalid timestamp"
        
        print("✓ JSON logging test passed!")
        print(f"✓ Created JSON file: {json_path}")
        print(f"✓ File contents: {json.dumps(loaded_data, indent=2)[:200]}...")


def main():
    """Run all tests."""
    print("🧪 Running JSON Logging Tests")
    print("=" * 40)
    
    test_json_logging()
    
    print()
    print("✅ All JSON logging tests passed!")


if __name__ == "__main__":
    main()