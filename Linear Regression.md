## Linear Regression
- Prediction function
  $$
  f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
  $$
- Learning $\mathbf{w}$ by minimizing the **mean square error**
- Method called Ordinary Least Squares (**OLS**)
  - The bias term is treated as a weight, with $w_0 = b$
  $$
  \mathbf{w}^* = \argmin \frac{1}{M} \sum_i (y^{(i)} -  (\mathbf{x}^{(i)})^T\mathbf{w})^2\\
   = \argmin \frac{1}{M} (\mathbf{y - Xw})^T(\mathbf{y-Xw})
  $$
- With taking derivative w.r.t. $\mathbf{w}$ and set to zero
  $$
  \mathbf{w}^* = \mathbf{(X^TX)^{-1}X^Ty}
  $$
- When there is **too many feature/data** use **gradient descent**
  - Mini-batch GD for large size of data