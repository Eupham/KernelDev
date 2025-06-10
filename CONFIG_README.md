# GPT Training Configuration Guide

This guide explains how to use the configuration system for training GPT models with configurable precision.

## Quick Start

### Using Default Configuration
```bash
python entry.py
```
This will use the default `config.yaml` file.

### Using a Custom Configuration File
```bash
python entry.py --config config_fast.yaml
```

### Overriding Specific Parameters
```bash
python entry.py --config config_fast.yaml --precision 32 --epochs 5 --batch-size 4
```

## Configuration Files

The training script supports YAML configuration files that define all training parameters. Command-line arguments take precedence over config file values.

### Available Configuration Files

1. **`config.yaml`** - Default balanced configuration (fp16, medium model)
2. **`config_fast.yaml`** - Fast training for development (fp16, small model)
3. **`config_quality.yaml`** - High-quality training (fp32, large model)
4. **`config_bf16.yaml`** - BF16 training for modern GPUs (bf16, balanced model)

## Configuration Structure

### Training Parameters
```yaml
training:
  precision: 32              # 16 for fp16/mixed precision, 32 for fp32, "bf16" for bfloat16/mixed precision
  epochs: 3                  # Number of training epochs
  learning_rate: 0.0003      # Learning rate
  batch_size: null           # Batch size (null = auto-estimate)
  weight_decay: 0.01         # Weight decay for regularization
  warmup_steps: 100          # LR warmup steps
  max_grad_norm: 1.0         # Gradient clipping threshold
  save_every: 500            # Save checkpoint every N steps
  eval_every: 200            # Evaluate every N steps
  log_every: 50              # Log every N steps
  checkpoint_dir: "checkpoints"
```

### Data Parameters
```yaml
data:
  dataset_name: "allenai/c4" # HuggingFace dataset name
  dataset_config: "en"       # Dataset configuration
  seq_len: 1024              # Sequence length for training
  max_samples: 5000          # Max samples from dataset
  max_eval_tokens: 50000     # Max tokens for evaluation
  num_workers: 0             # DataLoader workers
  shuffle_train: true        # Shuffle training data
```

### Model Parameters
```yaml
model:
  vocab_size: 256            # Vocabulary size (auto-updated)
  dim: 512                   # Model dimension
  n_layers: 12               # Number of transformer layers
  n_heads: 16                # Number of attention heads
  max_seq_len: 2048          # Maximum sequence length
  mlp_ratio: 4               # MLP expansion ratio
  causal: true               # Use causal attention
```

### Hardware Parameters
```yaml
hardware:
  available_memory_gb: 15    # Available GPU memory
  device: "auto"             # Device ("auto", "cuda", "cpu")
  memory_buffer_gb: 2        # Memory buffer reservation
```

## Command Line Arguments

All configuration parameters can be overridden via command line:

```bash
python entry.py \
  --config config_fast.yaml \
  --precision 16 \
  --batch-size 8 \
  --seq-len 512 \
  --epochs 2 \
  --learning-rate 0.001
```

### Available Arguments

- `--config`: Path to YAML configuration file (default: config.yaml)
- `--precision`: Floating point precision (16, 32, or "bf16")
- `--batch-size`: Override batch size
- `--seq-len`: Sequence length for training
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate

## Precision Modes

### FP32 (Full Precision)
```yaml
training:
  precision: 32
```
- Uses 32-bit floating point
- Higher memory usage
- Better numerical stability
- Recommended for final/production training

### FP16 (Mixed Precision)
```yaml
training:
  precision: 16
```
- Uses 16-bit floating point with automatic mixed precision
- Lower memory usage (allows larger batch sizes)
- Faster training on modern GPUs
- Gradient scaling for numerical stability
- Recommended for development and large models

### BF16 (Bfloat16 Mixed Precision)
```yaml
training:
  precision: "bf16"
```
- Uses bfloat16 with automatic mixed precision
- Similar memory usage to fp16
- Better numerical stability than fp16 (wider dynamic range)
- No gradient scaling typically needed
- Best for modern GPUs (A100, H100, RTX 30xx/40xx series)
- Recommended for production training on compatible hardware

## Memory Optimization

The system automatically estimates optimal batch sizes based on:
- Model size
- Sequence length
- Precision mode
- Available GPU memory

You can override the estimation by setting `batch_size` in the config or via command line.

## Examples

### Fast Development Training
```bash
python entry.py --config config_fast.yaml
```
- FP16 precision
- Small model (256 dim, 6 layers)
- Short sequences (512 tokens)
- Quick evaluation
- 2 epochs

### High-Quality Production Training
```bash
python entry.py --config config_quality.yaml
```
- FP32 precision
- Large model (768 dim, 16 layers)
- Long sequences (2048 tokens)
- Thorough evaluation
- 10 epochs

### BF16 Training for Modern GPUs
```bash
python entry.py --config config_bf16.yaml
```
- BF16 precision
- Balanced model (1024 dim, 12 layers)
- Good performance and stability
- 3 epochs

### Custom Training
```bash
python entry.py \
  --precision bf16 \
  --epochs 5 \
  --batch-size 12 \
  --learning-rate 0.0005
```

## Outputs

The training script generates:
- Model checkpoints in the specified checkpoint directory
- Training curves plot (`training_curves.png`)
- Training metrics (`training_metrics.pt`)
- Console logs with progress and results
- Sample text generation

## Tips

1. **For development**: Use `config_fast.yaml` with FP16 for quick iterations
2. **For production**: Use `config_quality.yaml` with FP32 for best results  
3. **For modern GPUs**: Use `config_bf16.yaml` with BF16 for balanced performance
4. **Memory issues**: Reduce batch size, sequence length, or model dimension
5. **Speed up training**: Use FP16 or BF16 precision and smaller models
6. **Better quality**: Use FP32 precision and larger models

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config or via `--batch-size`
- Use FP16 precision (`--precision 16`) or BF16 (`--precision bf16`)
- Reduce `seq_len` or model `dim`

### Slow Training
- Use FP16 or BF16 precision for speed
- Reduce evaluation frequency (`eval_every`)
- Use smaller model for development

### Poor Results
- Increase model size (`dim`, `n_layers`)
- Use FP32 precision for stability
- Increase training data (`max_samples`)
- Adjust learning rate
