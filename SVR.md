## Support Vector Regression
- Borrowed from SVM
- Forming a tube of the prediction function by the following optimization problem
  $$
  \begin{aligned}
    \min_{\mathbf{w}, b, \mathbf{\xi, \xi^*}} & \frac{1}{2}||\mathbf{w}||^2_2 + C\sum_i (\xi_i + \xi^*_i) \\
    \text{subject to } & y^{(i)} - w^Tx^{(i)} -b \le \epsilon + \xi_i \\
    & w^Tx^{(i)} + b - y^{(i)} \le \epsilon + \xi_i \\
    & \xi, \xi^* \succeq 0
  \end{aligned}
  $$