# Modal Deployment Guide for KernelDev

This guide explains how to run your KernelDev training on Modal's cloud GPUs.

## Prerequisites

1. Install Modal:
```bash
pip install modal
```

2. Set up Modal authentication:
```bash
modal setup
```

## Usage

### Running on Modal

The `modal_launch.py` file provides several commands for different training scenarios:

#### Default Training (H100)
```bash
modal run modal_launch.py
```
or
```bash
modal run modal_launch.py train
```

#### Training on A100 (for comparison)
```bash
modal run modal_launch.py a100
```

#### Run Inference Testing
```bash
modal run modal_launch.py inference
```

#### Run Precision Testing
```bash
modal run modal_launch.py precision
```

### Running Locally (for testing)

You can also test the launcher locally before deploying to Modal:

```bash
python modal_launch.py train
```

```bash
python modal_launch.py inference
```

## Configuration

The system uses two main configuration files:

- `config.yaml` - Default configuration for local training
- `config_modal.yaml` - Optimized configuration for Modal deployment

To use the Modal-optimized config, you can modify the launcher to use it, or create custom configurations as needed.

## GPU Options

### H100 (Default)
- Highest performance
- 32GB RAM allocation
- Best for large models and fast training

### A100 
- Good performance/cost balance
- 24GB RAM allocation  
- Good for most training scenarios

## Output and Results

All training outputs, checkpoints, and plots are automatically saved to a Modal volume named `kerneldev-vol`. This ensures your results persist between runs.

Results are saved to:
- `/data/results/` (for H100 runs)
- `/data/results_a100/` (for A100 runs)

## Monitoring

Modal provides a web interface where you can:
- Monitor training progress in real-time
- View logs and outputs
- Download results
- Manage costs

Access your Modal dashboard at: https://modal.com/

## Cost Optimization

- Use A100 for development and testing
- Use H100 for final training runs
- Adjust timeout values based on expected training time
- Consider using smaller models or shorter sequences for experimentation

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all required files are in the project directory
2. **CUDA Errors**: Check that your model fits in GPU memory
3. **Timeout**: Increase timeout value in the Modal function decorator
4. **Volume Issues**: Ensure the volume is properly mounted

### Debug Mode

Add debug prints to the launcher or entry.py to troubleshoot issues:

```python
print("Current directory:", os.getcwd())
print("Files available:", os.listdir("."))
print("Python path:", sys.path)
```

### Local Testing

Always test your code locally first:

```bash
python modal_launch.py train
```

This runs the same code path but locally, making debugging easier.

## Advanced Usage

### Custom Configurations

You can pass custom configurations by modifying the subprocess call in the Modal functions:

```python
result = subprocess.run([
    sys.executable, "entry.py", 
    "--config", "your_custom_config.yaml",
    "--precision", "16",
    "--batch_size", "16"
], ...)
```

### Multi-GPU Training

For multi-GPU training, modify the Modal function to use multiple GPUs:

```python
@app.function(
    gpu=modal.gpu.A100(count=2),  # Use 2 A100s
    volumes={"/data": vol}, 
    timeout=7200,
    image=image,
)
```

### Custom Images

You can extend the base image with additional packages:

```python
image = image.pip_install([
    "your-additional-package",
    "another-package"
])
```

## Example Workflow

1. **Development**: Test locally with `python modal_launch.py train`
2. **Initial Training**: Use A100 with `modal run modal_launch.py a100`  
3. **Production**: Scale up to H100 with `modal run modal_launch.py train`
4. **Evaluation**: Run tests with `modal run modal_launch.py inference`

This workflow helps optimize both development time and compute costs.
