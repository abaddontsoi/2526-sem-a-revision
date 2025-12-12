## SVM, Hard margin
- Find a hyperplane to seperate datapoints
- A maximum margin from the hyperplane and datapoints must be obtained
- The geometric margin is
  $$
    d^{(i)} = \frac{y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b)}{||\mathbf{w}||_2}
  $$
  the functional margin does not include the $||\mathbf{w}||_2$ term
- Since rescaling the $\mathbf{w}$ and $b$ term with any $\gamma$ is possible, then can set
  $$
  \min y(\mathbf{w}^T\mathbf{x}+b) = 1
  $$
- The opimization problem becomes
  $$
  \min \frac{1}{2}||\mathbf{w}||^2_2
  $$
  subject to $y^{(i)}(\mathbf{w}^T\mathbf{x^{(i)}}+b) \ge 1$
- Classification function
  - $y^* = \text{sign}(\mathbf{w}^T\mathbf{x}+b)$ points correctly classified

## Soft margin version
- Similar to hard margin, but introduce a **penalty** to points **inside margin** (points inside margin are allowed)
  $$
  \min \frac{1}{2}||\mathbf{w}||^2_2 + C\sum_i \xi_i
  $$
  subject to $y^{(i)}(\mathbf{w}^T\mathbf{x^{(i)}}+b) \ge 1 - \xi_i, \xi_i \ge 0$
- Smaller $C$ to allow more violations and vice versa

## Loss function
- Hinge Loss
  $$
  \sum_i \max(0, 1 - y^{(i)}(\mathbf{w}^T\mathbf{x^{(i)}}+b)) + \frac{1}{C}||\mathbf{w}||^2_2
  $$

## Multiclass SVM
- Use 1-vs-rest binary classifiers
- Pick the one with largest geometric margin
  $$
  y^* = \argmax_{c\in \{1, ..., C\}} (\mathbf{w}^T_c \mathbf{x^*}_c) /||\mathbf{w}_c||_2
  $$

## Pros
- Good for hight-dimensional features with low generalization error
- Well-defined probabilities

## Cons
- Only liear decision hyerplane