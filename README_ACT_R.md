# ACT-R Fan Effect Implementation

This document describes the implementation of ACT-R (Adaptive Control of Thought-Rational) declarative memory testing within the KernelDev cocktail party task framework.

## Overview

The ACT-R implementation tests the fundamental principle that higher "fan" (more associations connected to a concept) reduces activation, leading to slower and less accurate retrieval. This provides a cognitive theory-based evaluation of the model's memory and attention mechanisms.

## Theory Background

### ACT-R Declarative Memory
ACT-R models human declarative memory using an activation-based system where:

```
A_i = B_i + Σ_j W_j S_ji + ε
```

Where:
- `A_i`: activation of chunk i
- `B_i`: base-level activation  
- `W_j`: attentional weighting
- `S_ji`: strength of association
- `ε`: noise term

### Fan Effect
The fan effect predicts that as the number of associations (fan) for a concept increases, its activation decreases, resulting in:
- **Lower accuracy** for high-fan items
- **Smaller margins** (difference between correct and incorrect choice confidence)
- **Slower retrieval times** (in systems with time constraints)

## Implementation Details

### 1. Fan Calculation (`data_builder.py`)

#### Association Building
- **Window-based associations**: For each token, count distinct tokens that co-occur within a sliding window (default: 5 tokens)
- **Corpus-level statistics**: Build associations across the entire training corpus
- **Special token handling**: Skip special tokens (`[PAD]`, `[CLS]`, etc.) in association building

#### Fan Measurement
```python
def calculate_fan(self, anchor_token):
    """Fan = number of distinct associates that co-occur with the anchor."""
    return len(self.actr_association_table[anchor_token])
```

#### Anchor Selection
- **First content token**: Use the first non-special token in the true span as the anchor
- **Extensible**: Can be enhanced with more sophisticated anchor selection (e.g., head word, most frequent token)

### 2. Stimulus Construction

#### Cocktail Party Integration
The existing cocktail party task is enhanced with ACT-R metrics:
- **Fixed N**: Keep number of options constant (default: 4 - 1 gold + 3 distractors)
- **Fan stratification**: Trials can be categorized into low/medium/high fan tertiles
- **Similarity matching**: Control for similarity between gold and distractors

#### Trial Metadata
Each trial includes:
```python
{
    'anchor_token': token_id,
    'fan': fan_value,
    'sim_gold': 1.0,  # Self-similarity 
    'max_sim_distractor': max_similarity,
    'distractor_similarities': [sim1, sim2, sim3],
    'true_span_length': span_length,
    'num_distractors': 3
}
```

### 3. Metrics Logging (`train_loop.py`)

#### Per-Trial Logging
For each cocktail party trial, the system logs:
- **`fan(anchor)`**: Fan value of the anchor concept
- **`sim_gold`**: Similarity to gold span (always 1.0)
- **`max(sim_dj)`**: Maximum similarity to any distractor
- **`logits`**: Raw model outputs for all options
- **`choice`**: Model's predicted choice
- **`accuracy`**: Whether prediction was correct
- **`margin`**: Difference between correct and best incorrect logit

#### Batch Processing
```python
def _process_actr_metrics(self, actr_batch_metrics, scores, correct_idx):
    """Process ACT-R metrics for a batch and update trainer metrics."""
    # Calculate margins, probabilities, and accuracy
    # Store trial-level data for analysis
```

### 4. Analysis and Visualization

#### Fan Effect Analysis
```python
def analyze_actr_fan_effect(self):
    """Analyze ACT-R fan effect: higher fan should lead to lower accuracy/margin."""
    # Group trials by fan tertiles (low/medium/high)
    # Calculate mean accuracy and margin for each tertile
    # Return structured results for reporting
```

#### Expected Results
Based on ACT-R theory, we expect:
1. **Accuracy vs Fan**: Negative correlation (accuracy ↓ as fan ↑)
2. **Margin vs Fan**: Negative correlation (margin ↓ as fan ↑)
3. **Tertile Analysis**: Low fan > Medium fan > High fan (for both accuracy and margin)

## Usage

### 1. Standard Training with ACT-R
```bash
python entry.py --config config.yaml
```

The ACT-R functionality is automatically enabled when the `cocktail_party` task is configured.

### 2. Configuration
In your YAML config, ensure cocktail party task is enabled:
```yaml
tasks:
  cocktail_party:
    num_distractors: 3
    min_span_size: 10
    max_span_size: 50
```

### 3. Results Interpretation
During training, ACT-R metrics are logged and analyzed:
```
=== ACT-R Fan Effect Analysis ===
Total ACT-R trials analyzed: 1250
Fan range: [0, 45]

Fan Effect Results (expectation: accuracy ↓ as fan ↑):
  Low fan: accuracy=0.823, margin=2.145, trials=416
  Medium fan: accuracy=0.734, margin=1.892, trials=425  
  High fan: accuracy=0.651, margin=1.623, trials=409
```

## Model Fitting (Future Enhancement)

### ACT-R-Style Regression
The logged data enables ACT-R-style model fitting:
```
margin ~ γ0 + γ1·fan + γ2·sim_gold + γ3·max(sim_dj)
```

Expected coefficient signs:
- **γ1 < 0**: Negative fan effect (main prediction)
- **γ2 > 0**: Higher similarity to gold improves performance
- **γ3 < 0**: Higher similarity to distractors hurts performance

### Penalty Estimation
The coefficient γ1 provides a direct estimate of the fan penalty in the model's decision-making process.

## Files Modified

### `data_builder.py`
- Added ACT-R association table building
- Implemented fan calculation methods
- Enhanced cocktail party collation with ACT-R metrics
- Added similarity calculation utilities

### `train_loop.py`  
- Extended `TrainingMetrics` class with ACT-R tracking
- Added `analyze_actr_fan_effect()` method
- Implemented `_process_actr_metrics()` for trial logging
- Enhanced metric saving to include ACT-R data

### `entry.py`
- Integrated ACT-R corpus processing into training flow
- Added ACT-R analysis reporting after training completion
- Enabled automatic ACT-R analysis when cocktail party task is configured

## Theoretical Validation

This implementation provides a direct test of ACT-R's declarative memory principles:

1. **Ecological Validity**: Uses real corpus statistics for fan calculation
2. **Controlled Comparison**: Maintains fixed stimulus structure while varying fan
3. **Quantitative Analysis**: Provides numerical measures aligned with ACT-R theory
4. **Model Agnostic**: Tests any model architecture's conformity to cognitive principles

## Future Extensions

### Enhanced Analysis
- **Regression modeling**: Implement full ACT-R-style statistical models
- **Time-based metrics**: Add response time analysis for applicable architectures
- **Cross-validation**: Validate fan effects across different datasets

### Sophisticated Fan Calculation
- **Semantic similarity**: Use embedding-based association strength
- **Dependency parsing**: Build associations based on syntactic relationships  
- **Multi-scale windows**: Combine associations at different contextual scales

### Cognitive Architecture Integration
- **Working memory**: Model capacity constraints in the cocktail party task
- **Executive control**: Investigate attention allocation strategies
- **Learning dynamics**: Track how fan effects change during training

## References

1. Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press.
2. Anderson, J. R., & Lebiere, C. (1998). *The Atomic Components of Thought*. Lawrence Erlbaum Associates.
3. Bothell, D. (2017). ACT-R 7.x Reference Manual. Carnegie Mellon University.

This implementation bridges cognitive science theory with modern neural architectures, providing a principled framework for evaluating memory and attention mechanisms through the lens of human cognitive constraints.