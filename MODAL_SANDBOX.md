# Modal Sandbox for KernelDev

This directory contains a Modal sandbox script (`sandbox_run.py`) that allows you to run the KernelDev project in a cloud environment with GPU access.

## Prerequisites

1. Install Modal CLI:
```bash
pip install modal
```

2. Set up Modal authentication:
```bash
modal auth new
```

3. Ensure you have access to GPU resources in your Modal account (A100 or H100).

## Usage

### Basic Usage

Run the complete sandbox setup and validation:

```bash
python sandbox_run.py
```

This will:
- Create a Modal sandbox environment with A100 GPU
- Install all dependencies (PyTorch with CUDA, Triton, etc.)
- Clone the KernelDev repository
- Run comprehensive tests including:
  - CUDA and PyTorch validation
  - Attention behavior tests
  - Flash attention kernel demos
  - Training entry point validation
  - Quick training run to verify setup

### Interactive Usage

After running the script, you can use the sandbox interactively:

```python
import modal

# Connect to your running sandbox
app = modal.App.lookup("kerneldev-sandbox")
sb = modal.Sandbox.from_id("your-sandbox-id")  # Use ID from script output

# Run custom commands
sb.exec("bash", "-c", "cd /workspace/KernelDev && python test_demo.py").wait()

# Start an interactive shell
sb.exec("bash").wait()
```

### Customizing GPU Type

Edit the script to use different GPU types:

```python
# For H100 (if available)
gpu="H100"

# For multiple GPUs
gpu=modal.gpu.A100(count=2)

# For specific memory requirements  
gpu=modal.gpu.A100(memory=80)
```

## What the Script Tests

### 1. Environment Setup
- Python 3.11 with CUDA support
- PyTorch with CUDA 12.1
- Triton for GPU kernels
- All KernelDev dependencies

### 2. Repository Setup
- Clones KernelDev from the `remove-unnecessary-tasks` branch
- Verifies file structure and permissions

### 3. CUDA Validation
- Checks GPU availability and compute capability
- Validates PyTorch CUDA integration
- Reports GPU memory and specifications

### 4. Attention Behavior Tests
- Runs `test_attention_behaviors.py` with CUDA support
- Validates cocktail party attention patterns
- Tests hierarchical attention behaviors

### 5. Demo Scripts
- Executes `test_demo.py` for CUDA detection demos
- Runs `simulate_h100_test.py` for H100 simulation
- Shows before/after attention behavior comparisons

### 6. Training Validation
- Tests the main entry point (`entry.py`)
- Creates a minimal training configuration
- Runs a short training loop to validate setup

## Expected Output

The script provides comprehensive logging:

```
=== GPU Information ===
Device: NVIDIA A100-SXM4-80GB
Compute Capability: (8, 0)
Total Memory: 80.0 GB
✓ A100-optimized flash attention kernels will be used

=== Running Attention Behavior Tests ===
🖥️  GPU Detected: NVIDIA A100-SXM4-80GB
🔧 Compute Capability: 8.0
🚀 Hopper GPU (H100+): No
✅ 3 CUDA KERNEL TESTS PASSED!
Attention kernel behaviors are correctly implemented.

=== Modal Sandbox Setup Complete ===
🚀 Sandbox ID: sb-abc123def456
```

## GPU Requirements

- **Minimum**: A100 or equivalent with 40GB+ VRAM
- **Recommended**: A100 80GB or H100 for full feature testing
- **Compute Capability**: 8.0+ for optimal performance

## Cost Considerations

- A100 costs vary by provider and region
- The script runs for ~5-10 minutes for full validation
- Interactive usage extends cost based on duration
- Consider using spot instances if available

## Troubleshooting

### Common Issues

1. **Modal Authentication**:
   ```bash
   modal auth refresh
   ```

2. **GPU Unavailable**:
   - Check your Modal account GPU limits
   - Try different GPU types or regions
   - Use CPU for basic testing (modify script)

3. **Dependency Issues**:
   - The image build process handles most dependencies
   - Check Modal image build logs for errors

4. **Repository Access**:
   - Ensure the KernelDev repository is publicly accessible
   - Check branch name if using a different branch

### Debug Mode

Add debugging to the script:

```python
# Add verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check sandbox status
print(f"Sandbox status: {sb.object_id}")
print(f"Sandbox state: {sb.poll()}")
```

## Integration with Development Workflow

### Continuous Integration

Use the Modal sandbox in CI/CD:

```yaml
# .github/workflows/modal-test.yml
name: Modal GPU Tests
on: [push, pull_request]
jobs:
  gpu-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Modal
        run: pip install modal
      - name: Run GPU Tests
        run: python sandbox_run.py
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
```

### Development Testing

Quick development cycle:

```bash
# 1. Make changes locally
git add . && git commit -m "Update kernels"

# 2. Test in Modal
python sandbox_run.py

# 3. Interactive debugging if needed
modal shell kerneldev-sandbox
```

## Advanced Usage

### Custom Images

Create specialized images:

```python
custom_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("your-custom-packages")
    .pip_install("your-pip-packages")
    .run_commands("your-setup-commands")
)
```

### Volume Management

Persistent storage across runs:

```python
# Data persists between sandbox sessions
vol = modal.Volume.from_name("kerneldev-cache", create_if_missing=True)
volumes={"/data": vol, "/results": vol}
```

### Scheduling

Run tests on a schedule:

```python
@app.function(schedule=modal.Cron("0 2 * * *"))  # 2 AM daily
def nightly_tests():
    # Run comprehensive test suite
    pass
```