# KernelDev: Hierarchical Attention with Flash Attention Optimization

A specialized implementation of GPT models with hierarchical attention patterns for cocktail party tasks and efficient flash attention kernels.

## Overview

This repository implements a GPT model with two distinct attention modes:

1. **Teacher Forcing**: Standard causal attention for language modeling tasks
2. **Cocktail Party**: Hierarchical attention patterns for span-based reasoning tasks

The implementation features custom Triton kernels for memory-efficient flash attention with specialized attention masking capabilities.

## Architecture

### Core Components

#### `entry.py` - Main Entry Point
The primary execution script that orchestrates training and inference. Handles:
- Configuration loading from YAML files
- Multi-GPU distributed training setup
- Precision management (fp16, bf16, fp32)
- Memory estimation and batch size optimization
- Training loop coordination

**Key Features:**
- Automatic batch size estimation based on GPU memory
- Mixed precision training support
- Distributed training with process spawning
- Comprehensive GPU information reporting

#### `model.py` - GPT Model Architecture
Implements the core GPT transformer model with custom attention integration. Contains:
- `GPTModel`: Main transformer with embeddings, layers, and output projection
- `TransformerLayer`: Individual transformer blocks with attention and feed-forward
- `MultiHeadAttention`: Attention mechanism with flash attention kernel integration
- `SwiGLU`: Efficient activation function for feed-forward networks
- `RMSNorm`: RMS normalization for stable training

**Architecture Details:**
- Uses flash attention for memory efficiency
- Supports both causal and hierarchical attention patterns
- RMSNorm instead of LayerNorm for better numerical stability
- SwiGLU activation for improved performance

#### `original_kernel.py` - Specialized Flash Attention Implementation
The core flash attention kernel with hierarchical attention pattern support. This is the most complex component, implementing:

**Flash Attention Foundation:**
- Triton-based GPU kernels for memory-efficient attention computation
- Forward and backward pass implementations
- Block-wise computation to reduce memory usage from O(n²) to O(n)
- Support for different precisions (fp16, bf16, fp32)

**Hierarchical Attention Patterns:**
The kernel implements a 4-section hierarchical attention structure:

1. **Section 1 (Prefix)**: Bidirectional attention within prefix tokens (before and including `[CLS]`)
2. **Section 2 (Context)**: Causal attention within context + can attend to prefix section  
3. **Section 3 (Span Islands)**: Bidirectional within same span + can attend to context section
4. **Section 4 (Bridge/MASKQ)**: Bidirectional access to all spans + prefix (acts as aggregator)

**Attention Flow:**
- Hierarchical access: 4→(4+3), 3→(3+2), 2→(2_causal+1), 1→(1)
- Span isolation: Each `[SPAN]...[ES]` wrapper cannot attend to other spans
- Information flow: spans → context → prefix, with MASKQ aggregating from spans

**Incoherent Processing:**
- Optional Hadamard transform to reduce quantization error
- Automatic GPU capability detection (optimized for Hopper GPUs)
- Random sign generation for transform stability

**Key Functions:**
- `flash_attention()`: Main entry point for attention computation
- `_flash_attn_fwd()`: Forward pass Triton kernel
- `_flash_attn_bwd_dq()` & `_flash_attn_bwd_dkdv()`: Backward pass kernels
- `generate_hadamard_signs()` & `hadamard_transform()`: Incoherent processing utilities

#### `data_builder.py` - Data Processing and Task Management  
Handles dataset loading, tokenization, and task-specific data preparation. Manages:

**Dataset Management:**
- Multiple dataset loading strategies (C4, WikiText, fallback)
- UTF-8 byte-level tokenization
- Streaming dataset support for large corpora
- Train/validation/test splitting

**Task-Specific Processing:**
- **Teacher Forcing**: Standard sequence-to-sequence with `[CLS]` prefixing
- **Cocktail Party**: Complex span-based reasoning with metadata generation

**Cocktail Party Data Format:**
Input: `{prefix}[CLS]{context with [MASK]}[SPAN]option1[ES][SPAN]option2[ES]...[MASKQ]`

**Metadata Generation:**
- `in_span`: Boolean tensor marking tokens within span boundaries
- `span_id`: Integer tensor assigning unique IDs to each span (1-based, -1 for MASKQ)
- `is_prefix`: Boolean tensor marking prefix tokens (before and including `[CLS]`)

**Special Tokens:**
- `[PAD]`: Padding token (ID: 0)
- `[CLS]`: Classification/prefix separator (ID: 1)  
- `[MASK]`: Masked token in context (ID: 2)
- `[SPAN]`: Span start marker (ID: 3)
- `[ES]`: Span end marker (ID: 4)
- `[MASKQ]`: Query aggregator token (ID: 5)

#### `train_loop.py` - Training Infrastructure
Comprehensive training and evaluation framework with:

**Training Management:**
- Multi-task training coordination
- Gradient accumulation and clipping
- Learning rate scheduling with warmup
- Mixed precision training support
- Distributed training coordination

**Evaluation and Metrics:**
- Per-task loss computation
- Cocktail party accuracy measurement
- Training curve visualization
- Model checkpointing and restoration

**Inference and Generation:**
- Text generation with sampling controls (temperature, top-k, top-p)
- Task-specific inference modes
- Performance monitoring and logging

## Configuration

The system uses YAML configuration files for flexible parameter management:

- `config.yaml`: Default balanced configuration
- `config_fast.yaml`: Fast training for development  
- `config_bf16.yaml`: BFloat16 precision configuration
- `config_quality.yaml`: High-quality training settings

Configuration sections:
- `training`: Learning parameters, precision, batch size
- `data`: Dataset settings, sequence length, tokenization
- `model`: Architecture parameters (layers, heads, dimensions)
- `tasks`: Task-specific configurations
- `hardware`: GPU and memory settings
- `logging`: Output and monitoring controls

## Usage

### Basic Training
```bash
python entry.py                                    # Use default config
python entry.py --config config_fast.yaml         # Fast development config  
python entry.py --precision bf16 --batch-size 8   # Override specific settings
```

### Distributed Training
```bash
python entry.py --nproc_per_node 4  # 4-GPU training
```

### Tasks

#### Teacher Forcing
Standard language modeling with causal attention:
- Input format: `{task description}[CLS]{context}`
- Attention: Standard causal masking
- Objective: Next token prediction

#### Cocktail Party  
Span-based reasoning with hierarchical attention:
- Input format: `{prefix}[CLS]{context}[SPAN]option1[ES][SPAN]option2[ES]...[MASKQ]`
- Attention: Hierarchical with span isolation
- Objective: Span selection based on context

## Technical Details

### Flash Attention Implementation

The flash attention implementation maintains the mathematical equivalence to standard attention while reducing memory complexity:

**Standard Attention**: O(n²) memory for storing attention matrix
**Flash Attention**: O(n) memory using block-wise computation

**Key Optimizations:**
- Block-wise softmax computation with numerical stability
- Gradient checkpointing for backward pass
- Kernel auto-tuning for optimal tile sizes
- Mixed precision support with automatic scaling

### Memory Efficiency

The implementation includes several memory optimization strategies:
- Automatic batch size estimation based on available GPU memory
- Gradient accumulation for effective large batch training
- Mixed precision training to reduce memory usage
- Efficient tokenization and data loading

### GPU Compatibility

Optimized for modern NVIDIA GPUs:
- **T4**: Efficient kernels for development and small-scale training
- **A100**: High-performance kernels for large-scale training  
- **H100**: Cutting-edge optimizations for maximum throughput
- **Hopper**: Advanced incoherent processing features

## Status and Future Directions

### Current Flash Attention Status

The original_kernel.py maintains specialized flash attention capabilities with:
- ✅ Memory-efficient O(n) implementation
- ✅ Hierarchical attention pattern support
- ✅ Multi-precision compatibility
- ✅ Gradient computation support
- ✅ GPU-specific optimizations

### Potential Enhancements

While the current implementation is functionally complete, future improvements could include:

1. **Kernel Optimization**: Further tuning for specific GPU architectures
2. **Pattern Efficiency**: Optimized attention pattern computation for complex hierarchies
3. **Memory Scaling**: Enhanced memory management for very long sequences
4. **Precision Modes**: Additional numerical precision options
5. **Distributed Attention**: Cross-device attention computation for massive models

The implementation successfully maintains the core flash attention benefits while adding sophisticated attention pattern control for specialized tasks.

## Dependencies

- PyTorch (with CUDA support)
- Triton (for GPU kernel compilation)
- datasets (for data loading)
- numpy, matplotlib (for utilities and visualization)
- PyYAML (for configuration management)

## License

This project is designed for research and educational purposes in transformer architecture and flash attention optimization.