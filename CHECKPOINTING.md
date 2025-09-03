# Checkpointing System

This repository includes a robust checkpointing system that automatically saves and manages training progress, allowing for seamless training interruption and resumption.

## Features

### 🔄 Automatic Checkpoint Management
- **Periodic Saving**: Saves checkpoints at configurable intervals (default: every 1000 steps)
- **Rotation**: Maintains only N most recent checkpoints (default: 2), automatically removing older ones
- **Best Checkpoint**: Preserves `best_checkpoint.pt` separately from rotation

### 🎯 Complete State Preservation
- **Model State**: Full model parameters and weights
- **Optimizer State**: Optimizer parameters and internal state
- **Scheduler State**: Learning rate scheduler state
- **Dataset Position**: Current epoch and batch position for precise resumption
- **Training Metrics**: Loss history and performance metrics

### 🚀 Automatic Resume
- **Smart Detection**: Automatically detects existing checkpoints on startup
- **Seamless Resume**: Continues training from exact interruption point
- **Configurable**: Can be enabled/disabled via configuration

## Configuration

### YAML Configuration (config.yaml)
```yaml
training:
  # Checkpoint management
  auto_resume: true          # Enable automatic resume functionality
  max_checkpoints: 2         # Number of regular checkpoints to keep
  save_every: 1000          # Save checkpoint every N steps
  checkpoint_dir: "checkpoints"  # Directory for checkpoint storage
```

### Python Configuration
```python
from train_loop import TrainingConfig

training_config = TrainingConfig(
    auto_resume=True,           # Enable automatic resumption
    max_checkpoints=2,          # Keep only 2 most recent checkpoints  
    save_every=1000,           # Save every 1000 steps
    checkpoint_dir="checkpoints"
)
```

## Usage

### Starting Training
```bash
python entry.py --config config.yaml
```

The system will automatically:
1. Check for existing checkpoints in the checkpoint directory
2. Resume from the latest checkpoint if found
3. Start fresh training if no checkpoints exist

### Manual Checkpoint Control
```python
# In training code
trainer.save_checkpoint(step=1000, is_best=True)  # Save best checkpoint
trainer.save_checkpoint(step=1000)               # Save regular checkpoint

# Load specific checkpoint
loaded_step = trainer.load_checkpoint('checkpoints/checkpoint_step_1000.pt')
```

## File Structure

```
checkpoints/
├── checkpoint_step_1000.pt    # Most recent regular checkpoint
├── checkpoint_step_2000.pt    # Second most recent checkpoint
├── best_checkpoint.pt         # Best performing model (preserved)
├── training_metrics.pt        # Training history and metrics
└── inference_samples/         # Generated text samples
    └── inference_samples.json
```

## Checkpoint Contents

Each checkpoint file contains:
```python
{
    'step': 1000,                          # Training step number
    'model_state_dict': {...},             # Model parameters
    'optimizer_state_dict': {...},         # Optimizer state
    'scheduler_state_dict': {...},         # Learning rate scheduler
    'metrics': {...},                      # Training metrics
    'config': {...},                       # Training configuration
    'dataset_state': {                     # Dataset position tracking
        'current_epoch': 2,
        'current_batch': 50,
        'total_steps': 1000
    }
}
```

## Benefits

✅ **Space Efficient**: Only keeps N most recent checkpoints (default: 2)  
✅ **Automatic Cleanup**: Oldest checkpoints removed automatically  
✅ **Complete State**: Saves all necessary information for exact resumption  
✅ **Zero Configuration**: Works out-of-the-box with sensible defaults  
✅ **Configurable**: All aspects can be customized via config files  
✅ **Robust**: Handles interruptions gracefully with proper error handling  

## Example Workflow

1. **Start Training**: `python entry.py`
   - Creates `checkpoints/checkpoint_step_1000.pt`

2. **Continue Training**: 
   - Creates `checkpoints/checkpoint_step_2000.pt`
   - Removes old checkpoint automatically

3. **Training Interrupted**: Ctrl+C or system shutdown

4. **Resume Training**: `python entry.py`
   - Detects `checkpoint_step_2000.pt`
   - Resumes from step 2000 automatically
   - Continues with epoch/batch position restored

## Testing

Run the test suite to verify checkpointing functionality:

```bash
python test_checkpointing.py           # Basic functionality tests
python test_checkpoint_integration.py  # Integration tests
python demo_checkpointing.py          # Interactive demonstration
```

This checkpointing system ensures robust, resumable training with minimal storage overhead and zero manual intervention required.