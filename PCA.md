## Principal Component Analysis (PCA)
1. **Preprocess**: Center the data by subtracting the mean: μ = (1/M) ∑ x^{(i)}, then x^{(i)} -= μ.
2. **Covariance Matrix**: Σ = (1/M) X^T X.
3. **Eigendecomposition**: Find eigenvectors v1 to vN of Σ, sorted by eigenvalues σ1 ≥ ... ≥ σN (variance explained).
4. **Project**: For K dimensions, V = [v1 ... vK], Z = X V (low-dim representation).
5. **Reconstruct**: \hat{X} = Z V^T.
- Math simply: Maximize variance v^T Σ v subject to ||v||=1. Solution: v is eigenvector of Σ.

## Advantages
- Reduces dimensions while keeping most variance (info).
- Decorrelates features.
- Good for visualization (e.g., 2D/3D plots).
- Connected to SVD for efficiency.

## Disadvantages
- Assumes linear structure; fails on nonlinear data (e.g., curved manifolds).
- Sensitive to scaling—normalize data first.
- Orthogonal components may not capture all dependencies.

## Similarities and Differences with KPCA
- Similarities: Both reduce dimensions by maximizing variance; unsupervised; use eigenvectors.
- Differences: PCA is linear (hyperplanes); KPCA nonlinear via kernel trick (high-dim implicit map). PCA faster; KPCA handles curves but more compute.

## Examples
- Blob data: First PC along longest spread.
- Iris: 4D to 2D, preserves classes.
- Digits: 64D to 25D, explained variance plot to choose K.

## Connection to SVD
- PCA on covariance = SVD on X. Z = X V from PCA = U S from SVD (same subspace).
- When PCA fails: On data with higher-order dependencies or non-orthogonal axes.
