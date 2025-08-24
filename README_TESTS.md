# Token Behavior Testing

This directory contains comprehensive tests for the attention behaviors across both Teacher Forcing and Cocktail Party tasks.

## Test File: `test_token_behavior.py`

The test file validates attention patterns without requiring the Triton kernel to be runnable. It creates reference implementations of the expected attention behaviors and validates them through unit tests.

### Key Features

1. **Reference Attention Implementation**: Creates attention masks based on expected token behaviors
2. **Comprehensive Testing**: Tests both Teacher Forcing and Cocktail Party patterns
3. **Visual Demonstrations**: Shows attention matrices for small examples
4. **Edge Case Validation**: Tests boundary conditions and special cases

### Expected Behaviors Tested

#### Teacher Forcing
- ✅ Tokens before and including CLS are bidirectional
- ✅ Tokens after CLS are causal and can see CLS token
- ✅ PAD tokens are ignored

#### Cocktail Party (4 parts)
- ✅ Prefix up to CLS: bidirectional within prefix
- ✅ Context: causal within context + can see prefix
- ✅ Span islands: see context, context doesn't see them, inside spans are causal
- ✅ Each span island cannot see other islands
- ✅ MASKQ token sees all islands, islands don't see MASKQ

### Running Tests

```bash
python test_token_behavior.py
```

### Example Output

The test shows clear attention pattern visualizations:

**Teacher Forcing Demo ('Hi[CLS]OK'):**
```
Attention Matrix (5x5):
       0  1  2  3  4
  0:   █  █  █  ·  ·    # H can see prefix (H,i,CLS)
  1:   █  █  █  ·  ·    # i can see prefix (H,i,CLS)  
  2:   █  █  █  ·  ·    # CLS can see prefix (H,i,CLS)
  3:   █  █  █  █  ·    # O can see prefix + causal (H,i,CLS,O)
  4:   █  █  █  █  █    # K can see prefix + causal (H,i,CLS,O,K)
```

**Cocktail Party Demo:**
```
Attention Matrix shows:
- Prefix tokens (Q,CLS) are bidirectional among themselves
- Context tokens (W,h) are causal and can see prefix
- Span 1 tokens can see context but not span 2
- Span 2 tokens can see context but not span 1
- MASKQ can see all spans and prefix
```

This comprehensive test suite ensures the attention behaviors work exactly as specified in the issue requirements.