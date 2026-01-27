# Experiment Tracking Guide

This guide describes the experiment tracking system for managing and comparing training runs in KernelDev.

## Overview

The experiment tracking system provides:
- **Automatic hyperparameter logging**: All config values tracked automatically
- **Real-time metrics**: Training and validation losses, learning rates, etc.
- **Model versioning**: Automatic checkpoint versioning and artifact management
- **Run comparison**: Compare experiments across different configurations
- **Reproducibility**: Complete experiment history for reproducible research

## Supported Backends

- **Weights & Biases (W&B)**: Full-featured experiment tracking platform (recommended)
- **Local JSON**: Fallback option when W&B is not available

## Quick Start

### 1. Install Weights & Biases (Optional but Recommended)

```bash
pip install wandb
wandb login  # Follow prompts to authenticate
```

### 2. Enable in Configuration

Add to `config.yaml`:

```yaml
experiment_tracking:
  enable: true                    # Enable experiment tracking
  enable_wandb: true              # Use W&B (false for local-only)
  wandb_entity: your-team-name    # Optional: W&B team/user name
  project_name: kerneldev         # Project name in W&B
```

### 3. Run Training

```bash
python entry.py --config config.yaml
```

The system will automatically:
- Create a new experiment run
- Log all hyperparameters from config
- Track training metrics in real-time
- Save experiment log locally
- Provide W&B dashboard URL (if enabled)

## Configuration

### YAML Configuration

```yaml
experiment_tracking:
  # Enable/disable experiment tracking
  enable: true
  
  # Weights & Biases settings
  enable_wandb: true
  wandb_entity: your-username-or-team  # Optional
  project_name: kerneldev              # W&B project name
  
  # Experiment naming
  experiment_name: null  # Auto-generated if not provided
  tags: []               # List of tags (auto-generated if empty)
  
  # Logging frequency
  log_metrics_every: 1   # Log metrics every N steps (default: every step)
  log_model_every: 1000  # Log model checkpoints every N steps
```

### Environment Variables

Alternative to YAML config:

```bash
# W&B authentication
export WANDB_API_KEY=your-api-key

# Disable W&B (local logging only)
export WANDB_MODE=disabled

# Change project name
export WANDB_PROJECT=my-project

# Change entity
export WANDB_ENTITY=my-team
```

## Usage in Code

### Basic Usage

```python
from experiment_tracking import create_experiment_tracker

# Create tracker
tracker = create_experiment_tracker(
    config=config,
    enable=True,
    project_name="kerneldev",
    experiment_name="my-experiment",
    tags=["baseline", "bf16"]
)

# Log metrics
tracker.log_metrics({
    'train_loss': 0.5,
    'learning_rate': 0.0001
}, step=100)

# Log hyperparameters (additional to config)
tracker.log_hyperparameters({
    'effective_batch_size': 32
})

# Watch model for gradient tracking
tracker.watch_model(model, log_freq=100)

# Log model checkpoint as artifact
tracker.log_artifact(
    'checkpoints/checkpoint_step_1000.pt',
    artifact_type='model',
    name='checkpoint-step-1000'
)

# Finish experiment
tracker.finish()
```

### Integration with Training Loop

The experiment tracker is designed to integrate seamlessly with the existing training infrastructure:

```python
# In train_loop.py
class Trainer:
    def __init__(self, model, optimizer, data_builder, config, tracker=None):
        # ... existing init ...
        self.tracker = tracker
        
        # Watch model if tracker available
        if self.tracker:
            self.tracker.watch_model(model, log_freq=100)
    
    def train_step(self, batch):
        # ... existing training step ...
        
        # Log metrics
        if self.tracker and self.global_step % self.config.log_every == 0:
            self.tracker.log_metrics({
                'train_loss': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0],
                'step': self.global_step
            })
        
        return loss
```

## Metrics Logged

### Automatic Metrics

The system automatically logs:

**Training Metrics:**
- `train_loss`: Training loss per step
- `learning_rate`: Current learning rate
- `step`: Global training step
- `epoch`: Current epoch
- `grad_norm`: Gradient norm (if clipping enabled)

**Validation Metrics:**
- `val_loss`: Validation loss
- `val_accuracy`: Validation accuracy (task-specific)
- `cocktail_party_accuracy`: Cocktail party task accuracy

**System Metrics:**
- `step_time`: Time per training step
- `tokens_per_second`: Training throughput
- `memory_allocated`: GPU memory allocated
- `gpu_utilization`: GPU usage percentage

**Model Metrics:**
- Model parameters count
- Model size in MB
- Architecture details

### Custom Metrics

Log any custom metric:

```python
tracker.log_metrics({
    'custom_metric': value,
    'perplexity': math.exp(loss),
    'gradient_variance': grad_var
}, step=step)
```

## Comparing Experiments

### W&B Dashboard

Access the W&B dashboard to:
1. View real-time training curves
2. Compare multiple runs side-by-side
3. Create custom charts and reports
4. Download experiment data

Navigate to: `https://wandb.ai/{entity}/{project}`

### Local JSON Logs

Compare local experiments programmatically:

```python
import json
import matplotlib.pyplot as plt

# Load two experiments
with open('checkpoints/exp1/experiment_log.json', 'r') as f:
    exp1 = json.load(f)

with open('checkpoints/exp2/experiment_log.json', 'r') as f:
    exp2 = json.load(f)

# Extract losses
exp1_losses = [m['metrics'].get('train_loss') for m in exp1['metrics'] 
               if 'train_loss' in m['metrics']]
exp2_losses = [m['metrics'].get('train_loss') for m in exp2['metrics'] 
               if 'train_loss' in m['metrics']]

# Plot comparison
plt.plot(exp1_losses, label='Experiment 1')
plt.plot(exp2_losses, label='Experiment 2')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig('comparison.png')
```

## Experiment Organization

### Naming Convention

Auto-generated experiment names follow this pattern:
```
dim{model_dim}_layers{n_layers}_lr{learning_rate}
```

Example: `dim2048_layers12_lr3e-05`

Override with custom name:
```python
tracker = create_experiment_tracker(
    config=config,
    experiment_name="my-custom-experiment-v1"
)
```

### Tags

Use tags to organize experiments:

```python
tracker = create_experiment_tracker(
    config=config,
    tags=[
        "baseline",           # Experiment type
        "bf16",              # Precision
        "cocktail-party",    # Task
        "phase2",            # Development phase
        "v1.0"               # Version
    ]
)
```

Auto-generated tags include:
- Precision: `precision:bf16`, `precision:fp16`, `precision:fp32`
- Tasks: `task:teacher_forcing`, `task:cocktail_party`

### Notes

Add experiment notes:

```python
tracker = ExperimentTracker(
    config=config,
    notes="""
    Testing new attention pattern optimization.
    Expected to improve cocktail party accuracy by 5%.
    Baseline: 85% accuracy.
    """
)
```

## Best Practices

### 1. Always Enable Tracking

Enable tracking for all experiments, even quick tests:
```yaml
experiment_tracking:
  enable: true
```

### 2. Use Meaningful Names and Tags

```python
# Good
experiment_name="attention-optimization-v3"
tags=["optimization", "attention", "baseline-comparison"]

# Bad
experiment_name="test"
tags=["test"]
```

### 3. Log Consistently

Log metrics at consistent intervals:
```python
if step % config.log_every == 0:
    tracker.log_metrics(metrics, step=step)
```

### 4. Version Checkpoints

Log important checkpoints as artifacts:
```python
if step % config.save_every == 0:
    tracker.log_artifact(
        checkpoint_path,
        artifact_type='checkpoint',
        name=f'checkpoint-step-{step}'
    )
```

### 5. Document Experiments

Use notes field for context:
```python
tracker = ExperimentTracker(
    config=config,
    notes=f"""
    Hypothesis: Increasing warmup steps from 100 to 500 will improve convergence.
    Changes: warmup_steps: 100 -> 500
    Expected outcome: Faster convergence, lower final loss
    """
)
```

## Troubleshooting

### W&B Not Logging

**Problem**: Metrics not appearing in W&B dashboard

**Solutions**:
1. Check API key: `wandb login`
2. Verify internet connection
3. Check W&B status: https://status.wandb.ai/
4. Enable debug mode: `export WANDB_DEBUG=true`
5. Check logs for error messages

### Slow Performance

**Problem**: Training slows down with W&B enabled

**Solutions**:
1. Reduce logging frequency:
   ```yaml
   experiment_tracking:
     log_metrics_every: 10  # Log every 10 steps instead of every step
   ```
2. Disable model watching:
   ```python
   # Don't call tracker.watch_model()
   ```
3. Use offline mode, sync later:
   ```bash
   export WANDB_MODE=offline
   # Train...
   wandb sync checkpoints/wandb/latest-run
   ```

### Local Logs Too Large

**Problem**: `experiment_log.json` grows too large

**Solutions**:
1. Archive old experiments:
   ```bash
   mkdir -p archives
   mv checkpoints/experiment_log.json archives/exp_$(date +%Y%m%d).json
   ```
2. Use W&B instead of local logging for large experiments
3. Reduce logging frequency

### API Key Errors

**Problem**: `wandb.errors.UsageError: api_key not configured`

**Solutions**:
1. Login: `wandb login`
2. Set environment variable: `export WANDB_API_KEY=your-key`
3. Disable W&B: `export WANDB_MODE=disabled`

## Advanced Features

### Hyperparameter Sweeps

Use W&B sweeps for automated hyperparameter tuning:

```yaml
# sweep.yaml
program: entry.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  n_layers:
    values: [6, 12, 24]
  dim:
    values: [1024, 2048, 4096]
```

Run sweep:
```bash
wandb sweep sweep.yaml
wandb agent {sweep-id}
```

### Custom Dashboards

Create custom W&B dashboards:
1. Go to W&B project page
2. Click "Create new report"
3. Add custom charts:
   - Training curves comparison
   - Hyperparameter importance
   - System metrics correlation
4. Share with team

### Artifact Lineage

Track model lineage and dependencies:
```python
# Log training data as artifact
data_artifact = tracker.wandb.Artifact('training-data', type='dataset')
data_artifact.add_file('data/processed_data.pt')
tracker.wandb_run.log_artifact(data_artifact)

# Use artifact as input
data_artifact = tracker.wandb.use_artifact('training-data:latest')

# Log model with data lineage
model_artifact = tracker.wandb.Artifact('model', type='model')
model_artifact.add_file('checkpoints/best_model.pt')
tracker.wandb_run.log_artifact(model_artifact)
```

## Integration with CI/CD

Track benchmark experiments automatically:

```yaml
# .github/workflows/experiments.yml
name: Track Experiments

on:
  pull_request:
    branches: [main]

jobs:
  experiment:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run experiment
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python entry.py \
            --config config.yaml \
            --epochs 1
      
      - name: Comment results
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const log = JSON.parse(fs.readFileSync('checkpoints/experiment_log.json'));
            const finalLoss = log.metrics[log.metrics.length - 1].metrics.train_loss;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `🧪 Experiment complete!\nFinal training loss: ${finalLoss.toFixed(4)}`
            });
```

## Summary

The experiment tracking system provides:
- ✅ Zero-config tracking with sensible defaults
- ✅ W&B integration for powerful visualization and comparison
- ✅ Local JSON fallback for offline work
- ✅ Automatic hyperparameter and metric logging
- ✅ Model versioning and artifact management
- ✅ Reproducible research workflows

Enable it in your config and never lose track of an experiment again!
