## Rigde Regression
- Adding regularization term to Linear Regression
  - Prevent **overfitting**
  - Obtain numerically stable solutions
    - $X^TX$ not intervible
    - Enforce parameter space
- The regularization term will bias the least squares estimate
- Gives a smaller weight
  - More robust to perturbations of input
  - Better chance to zero out some redundant and uninformative features
- Ridge regression minimization objective function
  $$
  \min_\mathbf{w} \frac{1}{2}||\mathbf{Xw - y}||^2_2 + \frac{\alpha}{2}||\mathbf{w}||^2_2
  $$
- Closed-form solution
  $$
  \mathbf{w^*} = (\mathbf{X^TX} + \alpha\mathbf{I})^{-1}\mathbf{X^Ty}
  $$