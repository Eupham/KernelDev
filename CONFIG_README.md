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
  use_levenshtein_task: true       # Enable Levenshtein distance auxiliary task (default: true in provided configs).
  levenshtein_loss_weight: 0.1     # Weight for the Levenshtein auxiliary task loss (float).
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
# Note: When `use_levenshtein_task` is true, data is processed as individual sequences/sentences.
# Original and word-shuffled versions are created. `[CLS]` is prepended to sequences for the Levenshtein head.
# `[SEP]` tokens are generally not used for this task.
```

### Model Parameters
```yaml
model:
  vocab_size: 256            # Base vocabulary size. If `use_levenshtein_task` is true, `DataBuilder`
                             # effectively increases this by 1 for a special `[CLS]` token if `cls_token_id`
                             # (e.g. 256) is at or above this base size.
  dim: 512                   # Model dimension
  n_layers: 12               # Number of transformer layers
  n_heads: 16                # Number of attention heads
  max_seq_len: 2048          # Maximum sequence length
  mlp_ratio: 4               # MLP expansion ratio
  causal: true               # Use causal attention
  use_cls_prefix_attention: true   # For Levenshtein task, if CLS token's representation is used by the head,
                                 # enable its prefix attention (boolean).
  lm_self_critique_base_penalty: 0.3 # Base value added to LM loss before self-critique reward.
  lm_self_critique_reward_max: 0.3   # Max value for the self-critique reward scalar (0 to this value).
```

### Hardware Parameters
```yaml
hardware:
  available_memory_gb: 15    # Available GPU memory
  device: "auto"             # Device ("auto", "cuda", "cpu")
  memory_buffer_gb: 2        # Memory buffer reservation
  cpu_test_attention: false  # Force attention to CPU via fallback, bypassing Triton (boolean).
# If `cpu_test_attention: true`, all attention computations are forced onto the CPU
# using a Python-based fallback mechanism, bypassing the CUDA Triton kernels.
# This is useful for debugging attention logic or for running on systems without a
# compatible GPU. When true, the `device` will automatically be set to 'cpu' by the entry script.
# This mode is significantly slower than GPU execution.
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
- `--use-levenshtein-task <True/False>`: Enable/disable Levenshtein auxiliary task. Overrides config.
- `--levenshtein-loss-weight FLOAT`: Weight for Levenshtein auxiliary task loss component (e.g., 0.1). Overrides config.
- `--use-cls-prefix-attention <True/False>`: Enable/disable special prefix attention for CLS token if used by Levenshtein head. Overrides config.
- `--lm-self-critique-base-penalty FLOAT`: Base value added to LM loss before self-critique reward subtraction (e.g., 0.3). Overrides config.
- `--lm-self-critique-reward-max FLOAT`: Max value for the self-critique reward scalar (e.g., 0.3). Overrides config.
- `--cpu-test-attention`: Enable CPU attention fallback for testing (overrides config).

## Levenshtein Distance Auxiliary Task & Self-Critique LM Scaling

This section details the Levenshtein distance prediction auxiliary task and how it's used for a self-critique mechanism to scale the primary Language Modeling (LM) loss.

### Levenshtein Distance Task
- **Objective**: Train a model head (specifically, `model.levenshtein_head`) to predict the word-level Levenshtein distance.
- **Inputs to this head**: The head typically processes the representation of the `[CLS]` token, which is prepended to input sequences.
    - For original (unshuffled) sentences/sequences, the target Levenshtein distance is 0 (representing perfect coherence).
    - For word-shuffled versions of sentences/sequences, the target is the actual word-level Levenshtein distance between the original and shuffled word lists.
- **CLS Token Attention**: If `model: use_cls_prefix_attention: true` is set, the `[CLS]` token (when processed by the main transformer blocks, not just the head) receives prefix attention, allowing it to attend to all other tokens in its sequence, bypassing standard causal masking for this specific token. This helps it gather a global representation of the sequence for the Levenshtein head.
- **Loss Contribution**: The auxiliary task's loss is calculated as the Mean Squared Error (MSE) between the predicted distances and the target distances. This auxiliary loss is then weighted by `training: levenshtein_loss_weight` and added to the main LM loss.

### Self-Critique LM Loss Scaling
- **Objective**: Modulate the main LM loss for a given training item based on the Levenshtein head's assessment of the coherence of the LM's *own generated output* for that item. The intuition is to penalize the LM less if its own generation is coherent, and more if it's incoherent.
- **Process**:
    1. The standard per-item Cross-Entropy loss for Language Modeling (`per_item_lm_ce_loss`) is calculated based on the original, unshuffled input sequences.
    2. The model's own likely generated sequence (derived from `torch.argmax` of the LM logits from the original sentence) is created. This sequence is then prepended with a `[CLS]` token.
    3. This `[CLS]`-prepended, model-generated sequence is passed to the Levenshtein head to obtain a `d_self_critique` score. A lower score indicates better coherence (closer to 0, which is the ideal for an original sentence).
    4. The `d_self_critique` scores are normalized across the batch (`norm_d_item_batch`) so that the best (lowest distance) item in the batch gets a normalized score of 0, and the worst (highest distance) gets 1.
    5. A reward `r` is calculated: `r = (1.0 - norm_d_item_batch) * model: lm_self_critique_reward_max`. This reward is higher for more coherent generations (lower `norm_d_item_batch`).
    6. The final scaled LM loss for that training item is computed as: `scaled_lm_loss = (per_item_lm_ce_loss + model: lm_self_critique_base_penalty) - r`.
       - `lm_self_critique_base_penalty`: A base value added to the LM loss. This ensures that even with maximum reward, the LM loss doesn't become too small or negative, maintaining a learning signal.
       - `lm_self_critique_reward_max`: Scales the maximum possible reward.
- **Total Loss**: The mean of this `scaled_lm_loss` across the batch forms the LM component of the final combined loss, to which the weighted Levenshtein auxiliary loss is added.
- **Monitoring**: Training logs will include metrics such as the average `d_self_critique` score, its Exponential Moving Average (EMA), and the delta between the current score and its EMA. These help monitor the model's self-assessed coherence.

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
