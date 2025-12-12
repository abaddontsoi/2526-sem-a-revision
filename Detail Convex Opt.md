## Standard Form
$$
\begin{align}
\text{minimize}_{\mathbf{x}} f_0(\mathbf{x})\\
\text{subject to} f_i(\mathbf{x}) \le 0 \\
\mathbf{A}\mathbf{x} = \mathbf{b}
\end{align}
$$
- The optimal value $p^*$
  - Infeasible when $p^* \rightarrow \infty$
  - Unbounded when $p^* \rightarrow -\infty$
- Any local optimal is global optimal

## Lagrangian
- Method to solve (convex) optimization problem
- Inequality constrain must be convex
- Equality constrain must be affine (in $Ax = b$) form
- Forming Lagrangian
  $$
  L(\mathbf{x}, \mathbf{\lambda}, \mathbf{\nu}) = f_0(\mathbf{x}) + \sum_i \lambda_if_i(\mathbf{x}) + \sum_i \nu_i h_i(\mathbf{x})
  $$
- To perform minimization, take gradient of $L(\mathbf{x}, \mathbf{\lambda}, \mathbf{\nu})$ with respect to $\mathbf{x}$ and set to 0
  - Objective function can have multiple variable, differentiate to them all
- Solve all equations and present them in terms of $\lambda, \nu$

## Lagrangian Dual Problem
- The dual function
  $$
  \begin{align*}
  g(\mathbf{\lambda}, \mathbf{\nu}) = \inf_\mathbf{x} L(\mathbf{x}, \mathbf {\lambda}, \mathbf{\nu}) \\
  \text{subject to } \mathbf{\lambda} \succeq 0
  \end{align*}
  $$
  - Since all input variables are solved in terms of $\lambda, \nu$, no need to include them in dual function

## KKT Conditions
- Primal constraints: $f_i(\mathbf{x}) \le 0, h_i(\mathbf{x})=0$
- Dual constraints: $\mathbf{\lambda} \succeq 0$
- Complementary slackness: $\lambda_if_i(\mathbf{x}) = 0$
- Gradient of Lagrangian vanishes