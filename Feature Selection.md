## Feature Selection
- Goal: Select subset informative to class/target.
- In Lasso: $\min 1/2 ||\mathbf{Xw - y}||^2 + \alpha||\mathbf{w}||_1$; $\alpha$ controls sparsity (zeros).
- Select $\alpha$ for desired # features.
- Often supervised (uses labels).

## Advantages
- Simplifies models, reduces overfitting.
- Faster training/prediction.
- Interpretable (know which features matter).
- Handles irrelevant/noisy features.

## Disadvantages
- May miss interactions if not careful.
- Computation for exhaustive search high.
- Supervised needs labels.

## Similarities and Differences with Dimensionality Reduction
Similarities: Reduce features; preprocessing.
Differences: Feature sel subsets originals; dim red combines into new. Feature sel removes uninformative; dim red for correlations, unsupervised.

## Examples
- Boston housing: Lasso zeros TAX, RAD, etc.; keeps LSTAT, RM.
- Interpretation: Non-zero weights show important factors.

In regression: Lasso for sparse selection vs Ridge shrink all.
