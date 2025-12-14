## Gaussian Mixutre Model (GMM)
- Use to model elliptical cluster
- Use multivariate Gaussian by controlling covariance matrix
- Weighted sum of Gaussians
  $$
  p(\mathbf{x}) = \sum_j \pi_j \mathcal{N}(\mathbf{x}; \mathbf{\mu}_j, \mathbf{\Sigma}_j)
  $$
- Each Gaussian represents a elliptical cluster
  - Gaussian parameters are learnt by MLE
  $$
  \max_{\mathbf{\pi, \mu, \Sigma}} \sum_i \log \sum_j^K \pi_j \mathcal{N}(\mathbf{x}; \mathbf{\mu}_j, \mathbf{\Sigma}_j)
  $$
  - Learn faster by **Expectation Maximization**

## Expectation Maximization
- E-step + M-step
- E-step
  - Calculate cluster membership with soft assignment
  - Each data point can have fractional contribution to different clusters
  $$
  z_j^{(i)} = \frac{\pi_j \mathcal{N}(\mathbf{x}^{(i)}; \mathbf{\mu}_j, \mathbf{\Sigma}_j)}{\sum_k^K \pi_k \mathcal{N}(\mathbf{x}^{(i)}; \mathbf{\mu}_k, \mathbf{\Sigma}_k)}
  $$
- M-step
  - Update each Gaussian cluster's mean, covariance and weight using soft weighting
  - Soft count of points in cluster $j$: $M_j = \sum_i z^{(i)}_j$
  - Weight: $\pi_j = \frac{M_j}{M}$
  - Mean: $\mu_j = \frac{1}{M_j}\sum_i z^{(i)}_j\mathbf{x}^{(i)}$
  - Covariance: $\mathbf{\Sigma}_j = \frac{1}{M_j}\sum_i z^{(i)}_j(\mathbf{x}^{(i)} - \mu_j)(\mathbf{x}^{(i)} - \mu_j)^T$
- Cluster shape is affected by covariance matrix
  - $N\times N$ Matrix for $N$-dimensional data
  - Using $N$ parameters with diagonal matrix -> cluster align to coordinate axes
  $$
  \begin{bmatrix}
    a_{11} & 0 & 0 \\
    0 & ... & 0 \\
    0 & 0 & a_{nn} \\
  \end{bmatrix}
  $$
  - Using 1 parameter with diagonal matrix -> circle cluster
  $$
  \begin{bmatrix}
    a & 0 & 0 \\
    0 & ... & 0 \\
    0 & 0 & a \\
  \end{bmatrix}
  $$