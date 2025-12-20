# Loss Function Selection for Unbalanced Classification

## Problem Analysis

| Category | Classes | Samples/class | Total samples |
| -------- | ------- | ------------- | ------------- |
| Head     | 150     | 500+          | ~75,000+      |
| Tail     | 50      | 10-50         | ~500-2,500    |

**Imbalance ratio**: ~50:1 (worst case)

---

## Recommendation: ASL Can Help, But Consider Simpler Options First

For 200 classes with moderate imbalance, you have a few good options:

### Option 1: Class-Weighted NLL (Start Here)

```python
# Calculate inverse frequency weights
class_counts = torch.tensor([...])  # samples per class
weights = 1.0 / class_counts
weights = weights / weights.sum() * len(weights)  # normalize

criterion = nn.CrossEntropyLoss(weight=weights)
```

**Pros**: Simple, directly addresses imbalance, well-understood
**Cons**: Can overcorrect, making head classes worse

---

### Option 2: ASL (If Option 1 Underperforms on Tail)

```python
# For your problem, I'd tune gamma_neg lower
criterion = ASLSingleLabel(
    gamma_pos=0,    # keep at 0
    gamma_neg=2,    # lower than 4, you have fewer classes
    eps=0.1
)
```

**Why lower gamma_neg?** With only 200 classes (vs 7000), you have fewer "obvious" negatives to suppress. Too aggressive = underfitting.

---

### Option 3: Combine Both (Best for Your Case)

```python
class WeightedASL(nn.Module):
    def __init__(self, class_weights, gamma_neg=2, eps=0.1):
        super().__init__()
        self.class_weights = class_weights  # higher for rare classes
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, inputs, targets):
        # ASL computation...
        # Then multiply loss by class weight
        sample_weights = self.class_weights[targets]
        loss = (loss * sample_weights).mean()
        return loss
```

---

## Quick Comparison for Your Problem

| Method       | Head Classes (150) | Tail Classes (50) | Complexity |
| ------------ | ------------------ | ----------------- | ---------- |
| NLL          | Good               | Poor              | Simple     |
| Weighted NLL | Okay               | Better            | Simple     |
| ASL          | Good               | Better            | Medium     |
| Weighted ASL | Good               | Best              | Medium     |

---

## Suggested Approach: Progressive

```
1. Baseline:        NLL Loss -> measure tail class accuracy
2. If tail < 60%:   Add class weights
3. If still poor:   Switch to ASL (gamma_neg=2)
4. If still poor:   Weighted ASL + oversample tail classes
```

---

## Other Tricks That Help for 10-50 Sample Classes

| Technique                                         | Impact |
| ------------------------------------------------- | ------ |
| **Oversampling** tail classes (2-5x)              | High   |
| **Strong augmentation** on tail                   | High   |
| **Mixup/CutMix**                                  | Medium |
| **Temperature scaling** at inference              | Medium |
| **Two-stage training**: first balanced, then full | Medium |

---

## Practical Starting Point

```python
# Simple but effective for your case
from torch.utils.data import WeightedRandomSampler

# Oversample rare classes during training
sample_weights = 1.0 / class_counts[labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset))

# Use weighted CE
class_weights = (1.0 / class_counts) ** 0.5  # sqrt dampens extreme weights
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

The `** 0.5` (square root) prevents over-weighting rare classes too aggressively—a common practical trick.
