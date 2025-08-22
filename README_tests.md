# Attention Token Behavior Tests

This directory contains comprehensive tests for validating attention token behaviors in the KernelDev project.

## Test File: `test_attention_behaviors.py`

### Purpose
Tests the attention behaviors across both teacher forcing and cocktail party tasks without requiring Triton kernels to run. This makes the tests suitable for environments where Triton is not available.

### What it Tests

#### Teacher Forcing Task Behaviors:
- **Bidirectional Prefix**: Any token before CLS and including CLS are bidirectional
- **Causal Context**: All tokens after CLS should be causal and should see the CLS token
- **PAD Handling**: PAD tokens are ignored in attention calculations

#### Cocktail Party Task Behaviors (4-part structure):
1. **Prefix**: Bidirectional attention within prefix (tokens before/including CLS)
2. **Context**: Causal behavior, may include [MASK] token (no special behavior for mask)
3. **Span Islands**: `[SPAN]candidate text[ES]` structures where:
   - Spans see the context
   - Context does not see spans  
   - Inside span wrappers, tokens are causal
   - Each island cannot see another island
4. **MASKQ**: This token sees all the islands at the same time, spans should not see it

#### Special Token Behaviors:
- `[PAD]` (ID: 0): Ignored in attention calculations
- `[CLS]` (ID: 1): Serves as prefix/context boundary  
- `[MASK]` (ID: 2): Behaves as regular context token
- `[SPAN]` (ID: 3): Marks span start
- `[ES]` (ID: 4): Marks span end
- `[MASKQ]` (ID: 5): Global query token (span_id = -1)

### How to Run

```bash
# Run all tests
python test_attention_behaviors.py

# Run specific test case
python -m unittest test_attention_behaviors.AttentionBehaviorTests.test_teacher_forcing_attention_patterns

# Import as module for custom testing
python -c "import test_attention_behaviors; test_attention_behaviors.run_tests()"
```

### Test Categories

1. **`test_teacher_forcing_attention_patterns`**: Validates teacher forcing attention behaviors
2. **`test_cocktail_party_attention_patterns`**: Validates cocktail party attention behaviors  
3. **`test_special_token_behaviors`**: Tests special token handling
4. **`test_attention_mask_creation`**: Tests attention mask creation logic
5. **`test_attention_pattern_logic_validation`**: Validates the pattern detection logic
6. **`test_data_builder_cocktail_party_format`**: Tests data builder cocktail party format

### Dependencies

- `torch`: For tensor operations
- `numpy`: For numerical operations
- `datasets`: For data loading (HuggingFace datasets)
- `unittest`: For test framework (built-in)

### Expected Output

When all tests pass, you should see:
```
============================================================
✓ ALL TESTS PASSED!
Attention token behaviors are correctly implemented.
============================================================
```

### Notes

- Tests create synthetic data that matches expected format
- No actual Triton kernel execution required
- Tests validate the attention pattern logic without running inference
- Data builder tests may show warnings about fallback data when external datasets are unavailable
- All attention patterns are validated using mock attention scores that follow the expected behaviors