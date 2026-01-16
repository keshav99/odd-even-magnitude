# Teaching Neural Networks to Count: An Exploration in Digit Parity Detection
****Identifying even or odd number of digits in any whole number without using number of digits in features****


## Introduction

Can a neural network learn to count without being told how to count? This seemingly simple question masks a surprising depth of complexity. Consider the task: given any whole number, determine whether it contains an even or odd number of digits. To a human, this is trivial. We count the digits and check the parity. But for a machine learning model that only sees the number itself as a feature, the challenge becomes significantly more interesting.

> Q. If we were to design a neural network for identifying even and odd digit numbers without providing the length of the number as feature, and expect to learn the boundaries between each maagnitude for determining even and odd numbers for numbers outside the training range too, how would the network look like? Also importantly, how important is the supervision data for the model to learn this boundary function?

At first glance, this seems like a simple classification task: given a number, determine if it has an even or odd number of digits. The twist? We can't use the number of digits as a feature. The model must learn that the answer depends on which "magnitude bucket" the number falls into—specifically, the boundaries at powers of 10.

Mathematically, the number of digits in a number n is approximately $floor(log_{10}(|n|)) + 1$. For a neural network to solve this without explicit digit counts, it needs to learn these logarithmic boundaries and the alternating pattern: 1-9 (odd), 10-99 (even), 100-999 (odd), and so on.

## Hypothesis and Approach

We tested three distinct approaches to this problem: Gradient Boosted Trees (XGBoost), Random Forests, and a Multi-Layer Perceptron with Radial Basis Function (RBF) kernel preprocessing. Each brings different inductive biases to the table.

### Model Architectures

**Gradient Boosted Trees (XGBoost)**
- Sequential decision boundary refinement
- Hypothesis: Could learn power-of-10 threshold splits iteratively
- Configuration: max_depth=12, n_estimators=100, learning_rate=0.1

**Random Forest**
- Ensemble of independent decision trees
- Hypothesis: Could capture variance across orders of magnitude through bagging
- Configuration: n_estimators=100, max_depth=12, bootstrap=True

**MLP + RBF Features**
- Distance-based feature transformation followed by neural network
- Hypothesis: Explicit encoding of distances from powers of 10 (10¹, 10², ..., 10⁵) would make patterns linearly separable
- RBF kernel: exp(-γ × ||x - center||²) with γ = 10⁻¹⁰
- Architecture: 3 hidden layers (128, 64, 32 neurons) with ReLU activation

Gradient Boosted Trees build sequential decision boundaries, potentially learning to split at the critical power-of-10 thresholds where digit count changes (10, 100, 1000, and so on). Random Forests, with their ensemble of independent decision trees, might capture the variance across different orders of magnitude. The MLP with RBF features represents a different philosophical approach entirely: we explicitly provide the model with distance measures from key reference points (the powers of 10), allowing it to learn combinations of these spatial features.

The experimental design involved two dataset configurations. The first used balanced sampling across all orders of magnitude from 10⁰ to 10⁵, ensuring representative coverage of the problem space. The second employed random sampling across ranges, testing whether models could generalize beyond their training distribution.

## Results: The Surprising Dominance of Tree Methods

When trained on the balanced dataset, Random Forests achieved perfect accuracy, 100% on both validation and test sets. This isn't just impressive, it's revealing. Decision trees naturally excel at learning threshold-based rules, and this problem is fundamentally about thresholds. Each power of 10 represents a natural split point where the label flips. The Random Forest appears to have discovered and memorized these exact boundaries.

### Dataset 1: Balanced Sampling (10⁰ to 10⁵)

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| Random Forest | 100.00% | 100.00% |
| Gradient Boosted Trees | - | 99.51% |
| MLP + RBF Features | 99.94% | 95.96% |

Gradient Boosted Trees performed admirably with 99.51% test accuracy, though notably imperfect. The sequential refinement strategy of boosting may have introduced slight overfitting to specific patterns in the training data. When examining the failure cases, errors clustered around numbers with 5 and 7 digits, suggesting the model struggled with certain transitional ranges.

The MLP with RBF features, despite our careful feature engineering, achieved only 95.96% test accuracy. This is particularly interesting because we essentially handed the model the mathematical structure it needed to solve the problem. By computing distances from powers of 10, we transformed the input space into one where the solution should be more linearly separable. Yet it underperformed both tree-based methods significantly.

## The Generalization Challenge

The real test came when we evaluated models on numbers far outside their training range, spanning up to 10⁶⁰. Here, the results tell a sobering story about the nature of machine learning generalization.

### Out-of-Distribution Performance (10⁶ to 10⁶⁰)

**Models Trained on Dataset 1 (Balanced):**

| Model | Test Accuracy | Error Pattern |
|-------|---------------|---------------|
| Gradient Boosted Trees | 53.33% | Errors on 17 and 19-digit numbers |
| Random Forest | 16.67% | Systematic failure across all ranges |
| MLP + RBF Features | 60.00% | Errors on 17 and 19-digit numbers |

Random Forests collapsed to 16.67% accuracy, barely better than random guessing. Gradient Boosted Trees fared slightly better at 53.33%, while the MLP achieved 60.00%. These numbers might seem disappointing, but they reveal something fundamental about how these models work.

Tree-based methods learn explicit decision boundaries in the feature space they observe during training. When presented with numbers many orders of magnitude larger than anything seen before, they have no mechanism to extrapolate. A Random Forest that learned "if x < 100, label 0; if 100 ≤ x < 10000, label 1" simply doesn't know what to do with x = 10³⁴.

The MLP's marginally better performance suggests that the RBF transformation provides some degree of smoothness that aids extrapolation, though clearly not enough for reliable predictions. The distance-based features degrade gracefully rather than failing categorically.

## Dataset Composition and Its Consequences

Training on the randomly sampled dataset produced an intriguing inversion. Random Forests again achieved 100% validation accuracy, but now both tree methods performed identically poorly on out-of-distribution data (20% accuracy). The MLP dropped to 33.33%, worse than before.

### Dataset 2: Random Sampling (10⁰ to 10⁷)

**In-Distribution Performance:**

| Model | Validation Accuracy |
|-------|-------------------|
| Random Forest | 100.00% |
| Gradient Boosted Trees | 99.34% |
| MLP + RBF Features | 72.90% |

**Out-of-Distribution Performance (10⁶ to 10⁶⁰):**

| Model | Test Accuracy |
|-------|---------------|
| Random Forest | 20.00% |
| Gradient Boosted Trees | 20.00% |
| MLP + RBF Features | 33.33% |

This suggests that the sampling strategy profoundly affects what models learn. The random sampling likely created a less structured training set with uneven coverage across orders of magnitude. While tree methods could still memorize the training distribution perfectly, they had even less basis for extrapolation. The MLP, perhaps overfitting to the specific distance patterns in the random sample, lost even its marginal advantage.

Notice how the MLP's validation accuracy dropped from 99.94% to 72.90% under random sampling, while tree methods maintained near-perfect scores. This reveals a key difference: tree methods can perfectly partition any finite training set regardless of its structure, while the MLP's smooth decision boundaries struggle with irregular distributions even during training.

## Mathematical Intuition Behind the Failure

Why do these models struggle so profoundly with extrapolation? Consider what each architecture fundamentally learns. Tree-based methods partition the input space into regions and assign labels to each region. This works perfectly within the convex hull of the training data but provides no principled way to extend beyond it.

### Visualization of the Problem Space

The fundamental issue becomes clear when we visualize the decision boundaries. Tree models learn rules like:

```
if x < 10: return 0 (odd)
if 10 ≤ x < 100: return 1 (even)
if 100 ≤ x < 1000: return 0 (odd)
...
if 10000 ≤ x < 100000: return 1 (even)
else: ??? (never seen during training)
```

The MLP learns a composition of nonlinear transformations. Even with RBF features that encode relevant structure, the model optimizes to minimize training loss, not to discover the underlying mathematical law. It might learn that "numbers far from 10³ and close to 10⁴ tend to have an even digit count," but this doesn't generalize to "numbers far from 10³⁴ and close to 10³⁵."

What would work better? A model that could genuinely learn to compute logarithms, extract integer components, and check parity would generalize perfectly. But standard neural architectures don't spontaneously discover these compositional algorithms from pattern matching alone. This experiment highlights a fundamental limitation: interpolation masquerading as understanding.

## Conclusions

This exploration reveals both the power and limitations of modern machine learning approaches. Tree-based methods can achieve perfect accuracy on in-distribution data for this task, learning precise decision boundaries that capture the mathematical structure. Yet this success is illusory, collapsing entirely when asked to generalize beyond the training range.

The lesson extends beyond this particular problem. Many real-world applications involve extrapolation, whether predicting future trends, scaling to larger systems, or applying models in novel contexts. When we observe high validation accuracy, we must ask: has the model learned the underlying principle, or merely memorized the training distribution?

For problems with known mathematical structure, perhaps the answer isn't to train larger models on more data, but to build in the right inductive biases from the start. A hybrid approach that combines learned representations with symbolic computation might bridge the gap between pattern recognition and genuine understanding. The question isn't just whether machines can learn to count, but whether they can learn to count in ways that transfer beyond what they've seen.

