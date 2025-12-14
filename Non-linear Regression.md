## Non-linear Regression
- Can perform **kernel trick**
- An example is polynomial regression, the prediction function becomes
  $$
  \begin{align*}
  f(x) &= \sum_{i=0} w_i x^i \\
  &= \begin{bmatrix}
    w_0 & w_1 & ... & w_p
  \end{bmatrix}
  \begin{bmatrix}
    1 \\
    x \\
    x^2 \\
    \vdots \\ 
    x^p
  \end{bmatrix} \\ 
  & = \mathbf{w}^T \phi(x)
  \end{align*}
  $$
- Overfit may occur when degree increases
- Kernel Ridge Regression can become non-linear by 
  $$
  \begin{aligned}
    y^* &= (\mathbf{Xx}^*)^T(\mathbf{XX}^T + \alpha \mathbf{I}_M)^{-1}\mathbf{y} \\
    &= (\mathbf{k}^*)^T(\mathbf{K} + \alpha \mathbf{I}_M)^{-1}\mathbf{y} \\
  \end{aligned}
  $$
  with $\mathbf{K}$ is the $M \times M$ kernel matrix and $k_i^* = \mathcal{K}(\mathbf{x}^{(i)}, \mathbf{x}^*)$ between new point and all datapoints