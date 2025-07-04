# Multi-Task Training Implementation

## Overview

This implementation replaces the problematic self-critique mechanism with a robust multi-task training approach that eliminates the 25% performance overhead while providing meaningful training signals.

## System Architecture

### Combined Dataset (`CombinedMultiTaskDataset`)

The training system now uses a combined dataset that deterministically distributes three tasks within each batch for proper parallel training:

- **25% Levenshtein Task**: Word shuffling + distance prediction for text coherence
- **25% NSP Task**: Next Sentence Prediction with 3-class classification  
- **50% LM Task**: Standard autoregressive language modeling

### Task Types and Data Format

Each batch item contains:
```python
(input_tokens, lm_targets, auxiliary_value, task_type_flag)
```

Where `task_type_flag` indicates:
- `0.0`: Language Modeling task
- `1.0`: Levenshtein task  
- `2.0`: Next Sentence Prediction task

### Deterministic Batch Composition

The dataset ensures proper task distribution within each batch using a deterministic cyclic pattern:

**For default distribution (0.25, 0.25, 0.5):**
- Every 8 consecutive samples contain exactly:
  - 2 NSP samples (positions 0-1)
  - 2 Levenshtein samples (positions 2-3)  
  - 4 LM samples (positions 4-7)

**Benefits:**
- Eliminates random task selection that could create unbalanced batches
- Ensures consistent task distribution across all batches
- Maintains proper parallel training of all three tasks
- Supports custom task distributions while maintaining deterministic behavior

### Model Architecture Updates

**Extended Vocabulary:**
- CLS token: ID 256
- SEP token: ID 257  
- Vocabulary size: 258 (byte tokens + special tokens)

**New Model Heads:**
- Language modeling head (existing)
- Levenshtein head: Predicts normalized edit distance (1 output)
- NSP head: 3-class classification (3 outputs)

**Model Output Format:**
```python
logits, lm_loss, lev_distances, nsp_logits = model(input_tokens, targets)
```

### Task-Specific Processing

#### Levenshtein Task
- Input: `[CLS] shuffled_sentence`
- Target: Normalized Levenshtein distance (0.0 - 1.0)
- Loss: MSE between predicted and true distance
- LM targets: Masked (ignored in language modeling loss)

#### NSP Task
- Input: `[CLS] sentence_A [SEP] sentence_B [SEP]`
- Classes: 0=correct order, 1=reversed order, 2=both shuffled
- Loss: Cross-entropy classification loss
- LM targets: Masked (ignored in language modeling loss)

#### LM Task
- Input: `[CLS] normal_sentence`
- Standard autoregressive language modeling
- Both Levenshtein and NSP heads produce outputs but losses are ignored

### Training Loop

**Single Forward Pass:**
1. Model processes mixed batch
2. Task masks separate items by type
3. Loss calculation per task type:
   - LM loss: Calculated for LM and Levenshtein tasks only
   - Levenshtein loss: Calculated for Levenshtein tasks only  
   - NSP loss: Calculated for NSP tasks only

**Combined Loss:**
```python
total_loss = lm_loss + (lev_weight * lev_loss) + (nsp_weight * nsp_loss)
```

### Performance Benefits

1. **Eliminated 25% overhead**: Single forward pass vs. previous two-pass system
2. **Meaningful signals**: Each task provides useful training signal
3. **Simplified training**: No complex reward mechanisms or critique scoring
4. **Better convergence**: Multiple complementary objectives

### Configuration

No configuration changes required - the system automatically activates when `use_levenshtein_task: true` is set in the config.

Loss weights:
- Levenshtein: Configurable via `levenshtein_loss_weight` (default: 0.1)
- NSP: Fixed at 0.1 (can be made configurable if needed)

### Monitoring

Training logs now show:
```
Loss: X.XXXX, LM Comp: X.XXXX, Lev Aux: X.XXXX, NSP: X.XXXX, Pred Dist (orig): X.XXXX
```

All metrics are tracked and can be plotted for analysis.

## Usage

The system works transparently with existing configurations. Simply ensure:
1. `use_levenshtein_task: true` in config
2. Normal training command: `python entry.py`

The multi-task dataset will automatically be used and training will benefit from the improved approach.