# ACT-R Quick Start Guide

This guide provides a quick overview of how to use the ACT-R fan effect functionality implemented for the cocktail party task.

## What is ACT-R Fan Effect?

The ACT-R fan effect tests a core principle of human declarative memory: concepts with more associations (higher "fan") are harder to retrieve, leading to lower accuracy and smaller confidence margins.

**Theory**: `Higher Fan → Lower Activation → Worse Performance`

## Quick Usage

### 1. Enable ACT-R in Your Config

Ensure your YAML configuration includes the cocktail party task:

```yaml
tasks:
  cocktail_party:
    num_distractors: 3
    min_span_size: 10
    max_span_size: 50
```

### 2. Run Training

```bash
python entry.py --config your_config.yaml
```

ACT-R functionality is automatically enabled when cocktail party task is present.

### 3. View Results

After training, you'll see ACT-R analysis in the output:

```
=== ACT-R Fan Effect Analysis ===
Total ACT-R trials analyzed: 1250
Fan range: [0, 45]

Fan Effect Results (expectation: accuracy ↓ as fan ↑):
  Low fan: accuracy=0.823, margin=2.145, trials=416
  Medium fan: accuracy=0.734, margin=1.892, trials=425  
  High fan: accuracy=0.651, margin=1.623, trials=409
```

## What Gets Logged

For each cocktail party trial:
- **Fan**: Number of distinct associates for the anchor concept
- **Anchor**: The concept being tested (first content token in span)
- **Similarities**: Overlap between gold span and distractors
- **Accuracy**: Whether model chose correctly
- **Margin**: Confidence difference (correct logit - best incorrect logit)
- **Logits**: Raw model outputs for analysis

## Expected Results

If the model exhibits ACT-R-like behavior:
- ✅ **Low fan items**: Higher accuracy, larger margins
- ✅ **High fan items**: Lower accuracy, smaller margins
- ✅ **Gradual decline**: Performance decreases as fan increases

## Files Involved

- **`data_builder.py`**: Builds corpus associations, calculates fan values
- **`train_loop.py`**: Logs trial metrics, analyzes fan effects  
- **`entry.py`**: Orchestrates ACT-R corpus processing and reporting

## Testing

Run the test suite to verify functionality:

```bash
python test_actr_functionality.py
python demo_actr_integration.py
```

## Implementation Details

See `README_ACT_R.md` for complete technical documentation including:
- Theoretical background
- Implementation architecture
- Analysis methods
- Future extensions

## Key Metrics

- **Fan Calculation**: Sliding window co-occurrence counting
- **Association Window**: 5 tokens (configurable)
- **Tertile Analysis**: Groups trials by low/medium/high fan
- **Cognitive Validation**: Tests alignment with human memory principles

The ACT-R implementation provides a principled way to evaluate whether neural models exhibit human-like memory constraints and retrieval patterns.