# Kernel Principal Component Analysis (KPCA)

## Introduction
Hey there, newbie! KPCA is PCA's cooler sibling for nonlinear data. It maps data to a higher dimension where it's linear, then does PCA there—without actually computing the high-dim points, thanks to kernels!

From lecture: Extends PCA for data on non-flat surfaces.

## How It Works
1. **Kernel Matrix**: K_{ij} = K(x^{(i)}, x^{(j)}) = φ(x^{(i)})^T φ(x^{(j)}), where φ is implicit map.
2. **Center**: K' = (I - 1_M) K (I - 1_M), 1_M all 1/M.
3. **Eigendecompose K'**: Get eigenvectors w1 to wK, eigenvalues Mλ1 to MλK.
4. **Project new x*: z_k = ∑ w^{(i)}_k K(x*, x^{(i)}).

Kernel examples: Polynomial (curves), RBF (clusters).

## Advantages
- Captures nonlinear structures (e.g., separates clusters).
- Improves classification/regression as preprocessing.
- Kernel trick avoids explicit high-dim compute.

## Disadvantages
- Computationally expensive (O(M^3) for M samples).
- Kernel choice and params (e.g., σ in RBF) need tuning.
- No explicit basis; test needs all training data.

## Similarities and Differences with PCA
Similarities: Maximize variance; unsupervised; project data.
Differences: KPCA nonlinear (kernels); PCA linear. KPCA for curved data; PCA simpler/faster.

## Examples
- Polynomial kernel: Separates curved data, PC along curve.
- RBF: Splits into clusters; 2 PCs for 3 clusters.
- Digits: Better classification than PCA (logistic regression on coeffs).

Summary: KPCA for high-dim nonlinear via kernels; good for clustering too.
