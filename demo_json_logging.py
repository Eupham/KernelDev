#!/usr/bin/env python3
"""
Demonstration of JSON logging functionality for training metrics.
"""

import os
import tempfile
import json
import time
from pathlib import Path


def demo_json_logging():
    """Demonstrate the JSON logging functionality in action."""
    print("🎯 JSON Logging Demo for KernelDev Training")
    print("=" * 50)
    print()
    print("This demo shows how training metrics are now saved to JSON files:")
    print("• Periodic saving of all training metrics to JSON")
    print("• Configurable save frequency")
    print("• Timestamped entries for progress tracking")
    print("• Separate from existing inference samples")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        print(f"📁 Demo checkpoint directory: {checkpoint_dir}")
        print()
        
        # Simulate training configuration
        save_logs_json_every = 200  # Save every 200 steps
        total_steps = 1000
        
        print(f"⚙️ Configuration:")
        print(f"   - save_logs_json_every: {save_logs_json_every} steps")
        print(f"   - total_training_steps: {total_steps}")
        print()
        
        # Simulate training metrics
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'cocktail_party_metrics': [],
            'learning_rates': [],
            'step_times': [],
            'total_steps': 0,
            'best_val_loss': float('inf'),
            'best_step': 0
        }
        
        print("🚀 Starting Training Simulation...")
        print()
        
        for step in range(1, total_steps + 1):
            # Simulate realistic training metrics
            train_loss = max(0.1, 2.0 * (0.99 ** step))  # Exponential decay
            lr = 0.001 * (0.995 ** step)  # Learning rate decay
            step_time = 0.1 + (step % 20) * 0.005  # Variable step times
            
            # Update metrics
            metrics['train_losses'].append(train_loss)
            metrics['learning_rates'].append(lr)
            metrics['step_times'].append(step_time)
            metrics['total_steps'] = step
            
            # Validation every 100 steps
            if step % 100 == 0:
                val_loss = train_loss + 0.05  # Slightly higher than training loss
                metrics['val_losses'].append(val_loss)
                if val_loss < metrics['best_val_loss']:
                    metrics['best_val_loss'] = val_loss
                    metrics['best_step'] = step
                    
                # Add mock cocktail party metrics occasionally
                if step % 300 == 0:
                    metrics['cocktail_party_metrics'].append({
                        'accuracy': min(0.95, 0.5 + step * 0.0005),
                        'f1_score': min(0.93, 0.45 + step * 0.0005)
                    })
            
            # JSON logging every save_logs_json_every steps
            if step % save_logs_json_every == 0:
                # Create logs directory
                logs_dir = checkpoint_dir / "training_logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare data for JSON
                json_data = metrics.copy()
                json_data['timestamp'] = time.time()
                
                # Save to JSON
                json_path = logs_dir / "training_logs.json"
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Show progress
                avg_loss = sum(metrics['train_losses'][-50:]) / min(50, len(metrics['train_losses']))
                current_lr = metrics['learning_rates'][-1]
                
                print(f"📊 Step {step:4d} | Loss: {train_loss:.4f} (avg: {avg_loss:.4f}) | "
                      f"LR: {current_lr:.6f} | JSON Saved ✓")
        
        print()
        print("🎉 Training Simulation Complete!")
        print()
        
        # Show final results
        final_json_path = checkpoint_dir / "training_logs" / "training_logs.json"
        with open(final_json_path, 'r') as f:
            final_data = json.load(f)
        
        print("📈 Final Training Summary:")
        print(f"   - Total steps: {final_data['total_steps']}")
        print(f"   - Final train loss: {final_data['train_losses'][-1]:.4f}")
        print(f"   - Best validation loss: {final_data['best_val_loss']:.4f} (step {final_data['best_step']})")
        print(f"   - Training data points: {len(final_data['train_losses'])}")
        print(f"   - Validation data points: {len(final_data['val_losses'])}")
        print(f"   - Cocktail party evaluations: {len(final_data['cocktail_party_metrics'])}")
        print()
        
        # Show JSON structure
        print("📄 JSON File Structure:")
        print(f"   📁 {checkpoint_dir}/")
        print(f"   └── training_logs/")
        print(f"       └── training_logs.json")
        print()
        
        # Show sample JSON content
        print("📋 Sample JSON Content (first 20 lines):")
        with open(final_json_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:20], 1):
                print(f"   {i:2d}: {line.rstrip()}")
        if len(lines) > 20:
            print(f"   ... ({len(lines) - 20} more lines)")
        print()
        
        print("💡 Key Benefits:")
        print("   ✅ Human-readable training history in JSON format")
        print("   ✅ Easy integration with analysis tools and dashboards")
        print("   ✅ Configurable save frequency to balance storage vs. granularity")
        print("   ✅ Timestamped entries for temporal analysis")
        print("   ✅ Preserves all training metrics including cocktail party results")
        print("   ✅ Complements existing checkpoint and inference sample saving")
        print()
        
        print("🔧 Configuration in config.yaml:")
        print("   training:")
        print("     save_logs_json_every: 500  # Save JSON logs every N steps")
        print()
        
        print("🚀 Ready to use in production training!")


if __name__ == "__main__":
    demo_json_logging()