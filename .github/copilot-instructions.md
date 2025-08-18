# KernelDev: GPT Training with Flash Attention Kernels

**ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

This repository implements a GPT model training system with configurable precision (fp16, fp32, bf16) and custom Triton flash attention kernels for GPU acceleration.

## Working Effectively

### Bootstrap and Install Dependencies
1. Install Python dependencies:
   ```bash
   pip install torch numpy matplotlib pyyaml datasets transformers triton
   ```
   **Time**: 2-5 minutes. NEVER CANCEL. Set timeout to 600+ seconds.

2. Verify installation:
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```
   **Time**: 3-5 seconds.

### Core Functionality Testing
1. Test basic module imports:
   ```bash
   python -c "from model import GPTModel; from data_builder import DataBuilder; from train_loop import Trainer; print('✓ All modules imported successfully')"
   ```
   **Time**: 3-4 seconds.

2. Check CLI help and available options:
   ```bash
   python entry.py --help
   ```
   **Time**: <1 second.

3. Test data builder functionality:
   ```bash
   python data_builder.py
   ```
   **Time**: 2-3 seconds. Note: Will fall back to synthetic data due to network limitations.

### Training Configuration System
The system uses YAML configuration files with CLI overrides:

- **`config.yaml`**: Default balanced configuration (bf16, medium model)
- **`config_fast.yaml`**: Fast training for development (fp16, small model)
- **`config_quality.yaml`**: High-quality training (fp32, large model)  
- **`config_bf16.yaml`**: BF16 training for modern GPUs

### Basic Training Commands
**CRITICAL**: Training requires CUDA GPU. All commands will fail on CPU with "NotImplementedError: Could not run 'flash_attention::forward' with arguments from the 'CPU' backend."

For CUDA systems only:
```bash
# Fast development training
python entry.py --config config_fast.yaml

# High-quality production training  
python entry.py --config config_quality.yaml

# BF16 training for modern GPUs
python entry.py --config config_bf16.yaml

# Custom training with overrides
python entry.py --precision bf16 --epochs 5 --batch-size 12
```

**Training Time Estimates**: 
- Small model (config_fast): 5-15 minutes per epoch
- Medium model (config): 15-45 minutes per epoch  
- Large model (config_quality): 45-120 minutes per epoch
- **NEVER CANCEL training runs.** Set timeouts to 7200+ seconds (2+ hours).

## Test Suite

### Working Tests
1. **Module imports test** (3-4 seconds):
   ```bash
   python -c "from model import GPTModel; from data_builder import DataBuilder; from train_loop import Trainer; print('✓ All modules work')"
   ```

2. **Data builder test** (2-3 seconds):
   ```bash
   python data_builder.py
   ```
   Expected: Falls back to synthetic data, processes successfully.

### Failing Tests  
1. **Scheduler test** (fails but runs in 4 seconds):
   ```bash
   python test_scheduler.py
   ```
   **Known Issue**: Test expects precise floating point equality but gets small numerical differences. This is a test issue, not a functional problem.

2. **Scaling law test** (fails immediately):
   ```bash
   python scaling_law_test.py --config config.yaml --batch-size 2
   ```
   **Known Issue**: API mismatch in create_data_builder() function call.

3. **Empty test files**: 
   - `test_incoherent_default.py` (empty)
   - `test_quantization_benefit.py` (empty)  
   - `test_signs_theory.py` (empty)

## GPU Requirements and Limitations

### CUDA Requirement
- **Flash attention kernels require CUDA GPU**
- Training will fail on CPU with NotImplementedError
- Model uses custom Triton kernels for flash attention
- No CPU fallback is implemented

### GPU Compatibility
- **T4**: Good fp16 speedup, limited bf16 support
- **V100**: Excellent fp16 performance, limited bf16 support
- **A100/H100**: Best bf16 and fp16 performance, bf16 recommended
- **RTX 30xx/40xx**: Excellent bf16 and fp16 performance

## Precision Modes and Performance

### FP16 (Mixed Precision)
```bash
python entry.py --precision 16
```
- 50% memory reduction
- Faster on modern GPUs with Tensor Cores
- Automatic gradient scaling

### FP32 (Full Precision)  
```bash
python entry.py --precision 32
```
- Maximum stability and accuracy
- Higher memory usage
- Slower training

### BF16 (Bfloat16 Mixed Precision)
```bash
python entry.py --precision bf16
```
- 50% memory reduction (same as fp16)
- Better numerical stability than fp16
- Best for modern GPUs (A100, H100, RTX 30xx/40xx)

## Common Tasks and File Locations

### Key Project Files
- **Entry point**: `entry.py` - Main training script
- **Model**: `model.py` - GPT model with flash attention
- **Data**: `data_builder.py` - Dataset loading and tokenization
- **Training**: `train_loop.py` - Training loop implementation
- **Kernels**: `original_kernel.py` - Custom Triton flash attention kernels

### Configuration Files
```bash
ls *.yaml
```
Output:
```
config.yaml          # Default configuration
config_bf16.yaml     # BF16 optimized
config_fast.yaml     # Fast development
config_quality.yaml  # High quality/production
```

### Documentation Files  
```bash
ls *.md
```
Output:
```
CONFIG_README.md     # Configuration system guide
PRECISION_GUIDE.md   # Precision modes documentation
METRICS.md          # Training metrics documentation
```

## Validation and Debugging

### Always Run Before Committing
1. **Import test** (4 seconds):
   ```bash
   python -c "from model import GPTModel; from data_builder import DataBuilder; from train_loop import Trainer; print('✓ Imports work')"
   ```

2. **Data loading test** (3 seconds):
   ```bash
   python data_builder.py
   ```

3. **CLI validation** (1 second):
   ```bash
   python entry.py --help
   ```

### Memory and Performance Debugging
1. **Check GPU availability**:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
   ```

2. **Memory estimation test**:
   ```bash
   python -c "
   from entry import load_config, merge_config_with_args
   import argparse
   config = load_config('config_fast.yaml')
   args = argparse.Namespace(config='config_fast.yaml', precision=None, batch_size=None, seq_len=None, epochs=None, learning_rate=None)
   config = merge_config_with_args(config, args)
   print('Config loaded:', bool(config))
   "
   ```

## Troubleshooting Common Issues

### Out of Memory
- Reduce `--batch-size` parameter
- Use fp16 precision: `--precision 16`
- Use smaller model: `--config config_fast.yaml`
- Reduce sequence length: `--seq-len 512`

### CUDA Errors
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Training requires GPU - will not work on CPU
- Flash attention kernels need CUDA backend

### Dataset Loading Failures
- Expected behavior: Falls back to synthetic data when network unavailable
- Message "All primary dataset loading methods failed" is normal in offline environments
- System continues with fallback data for testing

### Network Connectivity Issues
- HuggingFace dataset downloads may fail in restricted environments
- System automatically falls back to synthetic text data
- This is expected behavior, not an error

## Expected Timing for Operations

| Operation | Time | Timeout Setting |
|-----------|------|----------------|
| Dependencies installation | 2-5 minutes | 600+ seconds |
| Module imports | 3-4 seconds | 30 seconds |
| Data builder test | 2-3 seconds | 30 seconds |
| CLI help | <1 second | 10 seconds |
| Config loading | <1 second | 10 seconds |
| Small model training (1 epoch) | 5-15 minutes | 1800+ seconds |
| Medium model training (1 epoch) | 15-45 minutes | 3600+ seconds |
| Large model training (1 epoch) | 45-120 minutes | 7200+ seconds |

**CRITICAL**: NEVER CANCEL long-running training operations. Flash attention training can take 2+ hours for large models. Always set generous timeouts and wait for completion.

## Repository Structure
```
├── entry.py              # Main training entry point
├── model.py              # GPT model implementation
├── data_builder.py       # Dataset loading and tokenization
├── train_loop.py         # Training loop implementation
├── original_kernel.py    # Custom Triton flash attention kernels
├── scaling_law_test.py   # Scaling law experiments (broken)
├── test_scheduler.py     # Scheduler tests (failing)
├── config*.yaml          # Configuration files
└── *.md                  # Documentation files
```

This system is designed for GPU-accelerated transformer training with custom kernels and will not function on CPU-only environments.