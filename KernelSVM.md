## Kernel SVM
- Dirive to dual form first, the partial derivatives of Lagrangian
  $$
  \begin{aligned}
    \nabla_w L = w - \sum_i \alpha_i y^{(i)} x^{(i)} = 0 \\
    \nabla_b L = \sum_i \alpha_i y^{(i)} = 0 \\
    \nabla_{\xi_i} L = C - \alpha_i - \beta_i = 0 \\
  \end{aligned}
  $$
- Optimizing Dual function
  $$
  \max_{\mathbf{\alpha}} g(\mathbf{\alpha}) = \sum_i \alpha_i - \frac{1}{2}\sum_{i, j = 1} y^{(i)} y^{(j)} \alpha_i \alpha_j (\mathbf{x}^{(i)})^T\mathbf{x}^{(j)}
  $$
  $$
  \begin{aligned}
    \text{subject to } \sum_i \alpha_i y^{(i)} = 0, \\
    0 \le \alpha_i \le C
  \end{aligned}
  $$
- Classification function can change to
  $$
  y^* = \text{sign}(\mathbf{w}^T\mathbf{x}^* + b) = \text{sign}(\sum_i \alpha_i y^{(i)}(\mathbf{x}^{(i)})^T\mathbf{x}^* + b)
  $$
- Use a kernel trick
  $$
  \mathbf{z} = \phi (\mathbf{x})
  $$
  where $\mathbf{z}$ has higher dimension than $\mathbf{x}$
  - For example, $\phi(\mathbf{x}) = [x_1, x_2, x_1^2]^T$
- Observed from the dual problem, there exists a term $(\mathbf{x}^{(i)})^T\mathbf{x}^{(j)}$, which then can change to $\phi(\mathbf{x}^{(i)})^T\phi(\mathbf{x}^{(j)})$
  - Hence a kernel function can befined as $\mathcal{K}(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) =\phi(\mathbf{x}^{(i)})^T\phi(\mathbf{x}^{(j)})$
  - This can save computational time
- Final Kernel SVM form
  $$
  \max_{\mathbf{\alpha}} g(\mathbf{\alpha}) = \sum_i \alpha_i - \frac{1}{2}\sum_{i, j = 1} y^{(i)} y^{(j)} \alpha_i \alpha_j \mathcal{K}(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})
  $$
  $$
  \begin{aligned}
    \text{subject to } \sum_i \alpha_i y^{(i)} = 0, \\
    0 \le \alpha_i \le C
  \end{aligned}
  $$
  - With prediction function
  $$
  y^* = \text{sign}(\sum_i \alpha_i y^{(i)}\mathcal{K}(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) + b)
  $$

## Pros and Cons
- Pros
  - Can have non-linear decision boundary
- Cons
  - Sensitive to kernel function and violation penalty $C$
  - Computationally expensive for cross-val
  - Required to compute and store a large the **kernel matrix** when dataset is **large**
    - A $M\times M$ matrix to store all combinations of values
    - `K[i][j]` = $\mathcal{K}(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$