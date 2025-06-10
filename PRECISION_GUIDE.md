# GPT Model Training with Configurable Precision

This project now supports fp32 (full precision), fp16 (mixed precision), and bf16 (bfloat16 mixed precision) training modes for improved performance and memory efficiency.

## Precision Options

### FP32 (Full Precision) - Default
- Uses 32-bit floating point for all operations
- Maximum accuracy and stability
- Higher memory usage
- Slower training

### FP16 (Mixed Precision)
- Uses 16-bit floating point for forward pass and gradients
- Maintains 32-bit precision for critical operations (loss scaling, optimizer states)
- Reduced memory usage (~50% reduction)
- Faster training on modern GPUs with Tensor Cores
- Automatic loss scaling to prevent gradient underflow

### BF16 (Bfloat16 Mixed Precision)
- Uses bfloat16 (16-bit brain floating point) for forward pass and gradients
- Maintains 32-bit precision for critical operations (optimizer states)
- Reduced memory usage (~50% reduction, same as fp16)
- Better numerical stability than fp16 due to wider dynamic range
- Typically doesn't require gradient scaling
- Faster training on modern GPUs (A100, H100, RTX 30xx/40xx series)
- Best balance of speed, memory efficiency, and stability

## Usage Examples

### Basic Usage
```bash
# Default fp32 training
python entry.py

# fp16 mixed precision training
python entry.py --precision 16

# bf16 mixed precision training
python entry.py --precision bf16

# fp32 explicit
python entry.py --precision 32
```

### Advanced Configuration
```bash
# bf16 with custom batch size and sequence length
python entry.py --precision bf16 --batch-size 32 --seq-len 2048

# bf16 with more epochs and custom learning rate
python entry.py --precision bf16 --epochs 10 --learning-rate 5e-4

# fp16 with custom configuration
python entry.py --precision 16 --batch-size 16 --seq-len 1024

# fp32 with reduced sequence length for testing
python entry.py --precision 32 --seq-len 512 --epochs 1
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--precision` | str | 32 | Floating point precision (16, 32, or "bf16") |
| `--batch-size` | int | auto | Override batch size estimation |
| `--seq-len` | int | 1024 | Sequence length for training |
| `--epochs` | int | 3 | Number of training epochs |
| `--learning-rate` | float | 3e-4 | Learning rate |

## Performance Considerations

### Memory Usage
- **FP16**: ~50% reduction in model parameter memory
- **FP16**: ~50% reduction in activation memory
- **FP16**: Gradients and optimizer states remain in fp32 for stability
- **BF16**: ~50% reduction in model parameter memory (same as fp16)
- **BF16**: ~50% reduction in activation memory (same as fp16)
- **BF16**: Gradients and optimizer states remain in fp32 for stability

### Speed
- **FP16**: Faster on GPUs with Tensor Cores (T4, V100, A100, RTX series)
- **FP16**: May be slower on older GPUs without Tensor Core support
- **FP16**: Automatic mixed precision handles precision switching
- **BF16**: Fastest on modern GPUs (A100, H100, RTX 30xx/40xx series)
- **BF16**: Better hardware support on newer architectures
- **BF16**: More stable training due to wider dynamic range

### Accuracy
- **FP16**: Minimal accuracy loss with proper loss scaling
- **FP16**: Automatic gradient scaling prevents underflow
- **BF16**: Better numerical stability than fp16
- **BF16**: Wider dynamic range reduces risk of overflow/underflow
- **BF16**: Often doesn't require gradient scaling
- **FP32**: Maximum precision for research or when stability is critical

## Technical Details

### Mixed Precision Implementation
- Uses PyTorch's Automatic Mixed Precision (AMP)
- `torch.amp.autocast('cuda')` for forward pass
- `torch.amp.GradScaler('cuda')` for gradient scaling
- Automatic loss scaling and unscaling

### Memory Estimation
The batch size estimation now accounts for precision:
- FP16: 2 bytes per parameter/activation
- FP32: 4 bytes per parameter/activation
- Optimizer states remain in fp32 regardless of model precision

### Flash Attention Compatibility
- Flash attention kernels work with both fp16 and fp32
- Kernel automatically adapts to input tensor dtype
- T4-optimized configurations support both precisions

## Testing

Run the precision test script to compare both modes:
```bash
python test_precision.py
```

This will:
1. Test both fp32 and fp16 training
2. Compare training times
3. Verify successful completion
4. Show performance differences

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory with fp16**
   - Even with fp16, you might need to reduce batch size on smaller GPUs
   - Use `--batch-size` to override automatic estimation

2. **NaN losses with fp16**
   - Very rare with automatic loss scaling
   - Try reducing learning rate if it occurs

3. **Slower fp16 performance**
   - Older GPUs may not benefit from fp16
   - T4 and newer GPUs should show speedup

### GPU Compatibility
- **T4**: Good fp16 speedup with Tensor Cores
- **V100**: Excellent fp16 performance
- **A100/H100**: Best fp16 performance
- **Older GPUs**: May prefer fp32

## Example Output

### FP32 Training
```
=== Setting up Precision ===
Using full precision training (fp32)...
✓ Model using fp32 precision
Parameter dtype: torch.float32
```

### FP16 Training
```
=== Setting up Precision ===
Setting up mixed precision training (fp16)...
✓ Model converted to fp16
✓ Gradient scaler initialized for mixed precision
Parameter dtype: torch.float16
```
