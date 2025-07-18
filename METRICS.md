# Cocktail Party Task Evaluation Metrics

This document outlines the metrics used to evaluate the performance of the model on the "cocktail party" task. These metrics provide a detailed picture of how well the model can identify and isolate the correct span of text from a set of distractors.

## 1. Representing Your Outputs

Assume you have N candidate spans in a batch, and each span i covers Lᵢ token‑positions. Your model produces either:

- **Hard mask Mᵢ ∈ {0,1}<sup>Lᵢ</sup>**: 1 if token t in span i is selected, 0 otherwise.
- **Soft scores Sᵢ ∈ [0,1]<sup>Lᵢ</sup>**: e.g. after sigmoid or mined from a softmax over spans.

You also know the gold mask **Gᵢ ∈ {0,1}<sup>Lᵢ</sup>** (1 for every token in the correct span, 0 elsewhere).

## 2. Binary vs. Soft‑Mask Sparsity

### A. Hamming‑style Cleanliness (Hard Masks)

For each span *i*, we define:

- **IoU (Intersection over Union)**:

  $$
  IoU_i = \\frac{|M_i \\land G_i|}{|M_i \\lor G_i|}
  $$

  - **Interpretation**: IoU ≈ 1.0 means near‑perfect one‑span selection.

- **Precision**:

  $$
  Precision_i = \\frac{|M_i \\land G_i|}{|M_i|}
  $$

  - **Interpretation**: Precision ≪ 1 implies “dirty” extra tokens are lit (the model is selecting more than just the correct span).

- **Recall**:

  $$
  Recall_i = \\frac{|M_i \\land G_i|}{|G_i|}
  $$

  - **Interpretation**: Recall ≪ 1 implies you’re missing parts of the gold span.

Where |A| counts the 1’s in mask A, and ∧,∨ are bitwise AND/OR. These metrics are aggregated over the batch (mean or weighted by |Gᵢ|) to get overall IoU/precision/recall.

### B. Soft‑Mask Overlap (Soft Scores)

When you have probabilities Sᵢ[t] ∈ [0,1], use a “soft” IoU:

$$
Soft‑IoU_i = \\frac{\\sum_t \\min(S_i[t], G_i[t])}{\\sum_t \\max(S_i[t], G_i[t])}
$$

This smoothly penalizes any non‑zero score on outside tokens and rewards high scores on the correct span.

## 3. Entropy & Gini: How Peaky Is Your Selection?

Sometimes you want to know how confident the model is within the chosen span:

### Token‑wise Entropy for span i:

$$
H_i = -\\frac{1}{L_i} \\sum_{t=1}^{L_i} [S_i[t] \\log S_i[t] + (1 - S_i[t]) \\log(1 - S_i[t])]
$$

- **Interpretation**: Low Hᵢ means “the model is very sure (0 or 1) at each token.”

### Gini Coefficient

The Gini Coefficient on the flattened vector `[Sᵢ, 1−Sᵢ]` also measures sparsity (0 = uniform; 1 = maximally peaky). This is not currently implemented but is a potential future metric.

## 4. Interpreting the Numbers

- **IoU ≈ 1 and entropy ≈ 0** → perfect, crisp one‑span pick (e.g. [0,0,0,1,1,1,0,…]).
- **IoU low, soft_IoU middling** → the model is “hedging” (soft picks) across candidates.
- **Precision low, recall high** → “dirty”—it’s lighting up too many tokens.
- **Precision high, recall low** → “conservative”—it picks only small parts of the span.

By tracking these five metrics, you’ll get a rich picture of how the model selects spans—far beyond just “accuracy.”
