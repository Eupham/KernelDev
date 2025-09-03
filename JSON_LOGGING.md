# JSON Logging for Training Metrics

## Overview

The KernelDev training system now supports automatic JSON logging of training metrics, making it easy to analyze training progress and integrate with external monitoring tools.

## Features

### 🔄 Automatic JSON Logging
- **Periodic Saving**: Saves complete training metrics to JSON at configurable intervals
- **Human-Readable Format**: JSON format for easy parsing and analysis
- **Timestamped Entries**: Each save includes timestamp for temporal tracking

### 📊 Complete Metrics Coverage
- **Training Losses**: All training step losses
- **Validation Losses**: Validation evaluation results
- **Learning Rates**: Learning rate schedule progression
- **Step Times**: Training step timing for performance analysis
- **Cocktail Party Metrics**: Task-specific evaluation metrics
- **Best Model Tracking**: Best validation loss and corresponding step

### 🎯 Complementary to Existing Systems
- **Inference Samples**: Existing inference sample JSON logging (separate file)
- **Checkpoints**: Standard PyTorch checkpoint saving continues unchanged
- **Console Logging**: Real-time logging continues as before

## Configuration

### YAML Configuration (config.yaml)
```yaml
training:
  # JSON logging frequency
  save_logs_json_every: 500  # Save training logs to JSON every N steps
  
  # Other logging settings
  log_every: 50              # Console logging frequency
  eval_every: 200           # Validation frequency
  save_every: 1000          # Checkpoint saving frequency
```

### Python Configuration
```python
from train_loop import TrainingConfig

training_config = TrainingConfig(
    save_logs_json_every=500,  # Save JSON logs every 500 steps
    log_every=50,             # Console logging frequency
    eval_every=200,           # Validation frequency
    save_every=1000           # Checkpoint frequency
)
```

## File Structure

```
checkpoints/
├── training_logs/
│   └── training_logs.json     # Complete training metrics
├── inference_samples/
│   └── inference_samples.json # Periodic inference samples
├── checkpoint_step_1000.pt    # Model checkpoints
├── checkpoint_step_2000.pt
├── best_checkpoint.pt         # Best model checkpoint
└── training_metrics.pt        # PyTorch metrics (legacy)
```

## JSON Structure

The `training_logs.json` file contains:

```json
{
  "train_losses": [0.5, 0.4, 0.3, ...],          // All training losses
  "val_losses": [0.35, 0.28, ...],               // Validation losses  
  "cocktail_party_metrics": [                     // Task-specific metrics
    {"accuracy": 0.85, "f1_score": 0.82},
    ...
  ],
  "learning_rates": [0.001, 0.0009, ...],        // Learning rate history
  "step_times": [0.1, 0.12, 0.11, ...],          // Step timing data
  "total_steps": 1500,                            // Current step count
  "best_val_loss": 0.25,                         // Best validation loss
  "best_step": 1200,                             // Step of best model
  "timestamp": 1642123456.789                    // Unix timestamp
}
```

## Usage Examples

### Analyzing Training Progress
```python
import json

# Load training logs
with open("checkpoints/training_logs/training_logs.json", 'r') as f:
    logs = json.load(f)

# Plot training curve
import matplotlib.pyplot as plt
plt.plot(logs['train_losses'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.show()

# Check best performance
print(f"Best validation loss: {logs['best_val_loss']} at step {logs['best_step']}")
```

### Integration with Monitoring Tools
```python
# Example: Send metrics to monitoring dashboard
import requests

with open("checkpoints/training_logs/training_logs.json", 'r') as f:
    logs = json.load(f)

# Send latest metrics to dashboard
current_loss = logs['train_losses'][-1]
current_lr = logs['learning_rates'][-1]
timestamp = logs['timestamp']

# Post to monitoring endpoint
requests.post("http://monitoring-dashboard/metrics", json={
    "loss": current_loss,
    "learning_rate": current_lr,
    "step": logs['total_steps'],
    "timestamp": timestamp
})
```

## Benefits

### ✅ **Easy Analysis**
- Standard JSON format works with any analysis tool
- No need to parse PyTorch checkpoint files
- Human-readable for quick inspection

### ✅ **External Integration**
- Simple to integrate with monitoring dashboards
- Compatible with data science workflows
- Easy to export to databases or visualization tools

### ✅ **Historical Tracking**
- Complete training history in one file
- Timestamped for temporal analysis
- Preserves all metric types

### ✅ **Configurable Granularity**
- Adjust save frequency based on needs
- Balance between storage usage and detail level
- No performance impact on training loop

## Migration from Existing System

The JSON logging is **additive** - it doesn't change existing functionality:

- ✅ Existing checkpoint saving continues unchanged
- ✅ Console logging continues as before  
- ✅ Inference samples still saved separately
- ✅ PyTorch metrics files still created
- ✅ No breaking changes to existing workflows

## Testing

Run the included tests to verify functionality:

```bash
# Test basic JSON logging
python test_json_integration.py

# Demo the functionality
python demo_json_logging.py

# Verify no regressions
python test_checkpointing.py
```

## Performance Impact

- **Minimal**: JSON saving happens only every N steps (default: 500)
- **Non-blocking**: File I/O doesn't impact training performance
- **Lightweight**: JSON serialization is fast for metric data
- **Optional**: Can be disabled by setting very high save frequency

## Ready for Production

The JSON logging system is production-ready and provides enhanced observability into training runs without impacting existing functionality.