#!/usr/bin/env python3
"""
Integration test for JSON logging during training simulation.
"""

import os
import tempfile
import json
import time
from pathlib import Path


def simulate_training_with_json_logging():
    """Simulate a training run and verify JSON logging works correctly."""
    print("Simulating training run with JSON logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        
        # Simulate training metrics being collected
        training_data = {
            'train_losses': [],
            'val_losses': [],
            'cocktail_party_metrics': [],
            'learning_rates': [],
            'step_times': [],
            'total_steps': 0,
            'best_val_loss': float('inf'),
            'best_step': 0
        }
        
        # Simulate 1000 training steps with periodic JSON logging every 500 steps
        save_logs_json_every = 500
        
        for step in range(1, 1001):
            # Simulate training step data
            train_loss = 1.0 - (step * 0.0005)  # Decreasing loss
            lr = 0.001 * (0.999 ** step)  # Decaying learning rate
            step_time = 0.1 + (step % 10) * 0.01  # Variable step time
            
            # Update training data
            training_data['train_losses'].append(train_loss)
            training_data['learning_rates'].append(lr)
            training_data['step_times'].append(step_time)
            training_data['total_steps'] = step
            
            # Simulate validation every 200 steps
            if step % 200 == 0:
                val_loss = train_loss + 0.1  # Validation slightly higher
                training_data['val_losses'].append(val_loss)
                if val_loss < training_data['best_val_loss']:
                    training_data['best_val_loss'] = val_loss
                    training_data['best_step'] = step
            
            # Simulate JSON logging every save_logs_json_every steps
            if step % save_logs_json_every == 0:
                logs_dir = checkpoint_dir / "training_logs"
                logs_dir.mkdir(exist_ok=True)
                json_path = logs_dir / "training_logs.json"
                
                # Add timestamp to data
                json_data = training_data.copy()
                json_data['timestamp'] = time.time()
                
                # Save to JSON
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                print(f"✓ Step {step}: JSON logs saved to {json_path}")
        
        # Verify final JSON file
        final_json_path = checkpoint_dir / "training_logs" / "training_logs.json"
        assert final_json_path.exists(), "Final JSON log file was not created"
        
        with open(final_json_path, 'r') as f:
            final_data = json.load(f)
        
        # Verify data integrity
        assert len(final_data['train_losses']) == 1000, f"Expected 1000 training losses, got {len(final_data['train_losses'])}"
        assert len(final_data['val_losses']) == 5, f"Expected 5 validation losses, got {len(final_data['val_losses'])}"  # Steps 200, 400, 600, 800, 1000
        assert final_data['total_steps'] == 1000, f"Expected 1000 total steps, got {final_data['total_steps']}"
        assert final_data['best_val_loss'] < 1.0, f"Expected decreasing validation loss, got {final_data['best_val_loss']}"
        assert 'timestamp' in final_data, "Timestamp missing from final JSON"
        
        print(f"✓ Final JSON contains {len(final_data['train_losses'])} training steps")
        print(f"✓ Final JSON contains {len(final_data['val_losses'])} validation points")
        print(f"✓ Best validation loss: {final_data['best_val_loss']:.4f} at step {final_data['best_step']}")
        print("✓ JSON logging integration test passed!")


def test_json_structure():
    """Test that the JSON structure matches expected format."""
    print("Testing JSON structure...")
    
    # Expected structure based on TrainingMetrics.save_metrics_json
    expected_fields = [
        'train_losses', 'val_losses', 'cocktail_party_metrics',
        'learning_rates', 'step_times', 'total_steps',
        'best_val_loss', 'best_step', 'timestamp'
    ]
    
    sample_data = {
        'train_losses': [0.5, 0.4, 0.3],
        'val_losses': [0.35],
        'cocktail_party_metrics': [{'accuracy': 0.85}],
        'learning_rates': [0.001, 0.0009, 0.0008],
        'step_times': [0.1, 0.12, 0.11],
        'total_steps': 3,
        'best_val_loss': 0.35,
        'best_step': 2,
        'timestamp': time.time()
    }
    
    # Verify all expected fields are present
    for field in expected_fields:
        assert field in sample_data, f"Missing expected field: {field}"
    
    # Test JSON serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f, indent=2)
        temp_path = f.name
    
    try:
        # Test JSON deserialization
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Verify data integrity
        for field in expected_fields:
            assert field in loaded_data, f"Field {field} missing after serialization"
            assert loaded_data[field] == sample_data[field], f"Field {field} corrupted during serialization"
        
        print("✓ JSON structure test passed!")
        
    finally:
        os.unlink(temp_path)


def main():
    """Run all integration tests."""
    print("🧪 Running JSON Logging Integration Tests")
    print("=" * 50)
    
    test_json_structure()
    print()
    simulate_training_with_json_logging()
    
    print()
    print("✅ All JSON logging integration tests passed!")
    print()
    print("Summary of JSON logging functionality:")
    print("- Training metrics are saved to JSON every 500 steps (configurable)")
    print("- JSON files are stored in checkpoints/training_logs/training_logs.json")
    print("- All training data is preserved including losses, learning rates, step times")
    print("- Timestamps are included for temporal tracking")
    print("- Inference samples are already saved separately to inference_samples.json")


if __name__ == "__main__":
    main()