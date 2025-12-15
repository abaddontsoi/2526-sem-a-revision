## Dimensionality Reduction

## Linear Case
- Project to lower-dim hyperplane (line/plane).
- Math: x^4{(i)} = ∑ z^{(i)}_k b_k (basis b, codes z).
- Matrix: X ≈ Z B + ε (noise).
- Learn: Minimize ||X - Z B||_F (Frobenius norm).
- Solve via ALS: Alternate optimize Z (fix B), B (fix Z).
- Optimal: SVD gives unique solution (Eckart-Young).

Reasons: Preprocess for speed, denoise, visualize.

## Advantages
- Reduces compute (curse of dimensionality).
- Denoises by project and reconstruct.
- Visualizes high-dim data.
- Unsupervised.

## Disadvantages
- May lose info if K too small.
- Linear assumes flat structure; nonlinear needs extensions.
- Not same as compression (entropy focus).

## Similarities and Differences with Feature Selection
Similarities: Both reduce features; preprocessing; make ML easier.
Differences: Dim red creates new combined features; feature sel picks subsets of originals. Dim red unsupervised, handles correlations; feature sel often supervised, removes uninformative.

## Examples
- Images: Approx as weighted basis images.
- Documents: Bag-of-words to topics.

Vs Data Compression: Dim red reduces dims, not necessarily code length.
