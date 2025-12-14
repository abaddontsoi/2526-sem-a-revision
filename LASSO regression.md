## Lasso Regression
- Some weight in **Ridge Regression** is small but non-zero
  - Better to zero out
- LASSO means "least absolute shrinkage and selection operator"
- The regularzation term become $||w||_1$
  $$
  \min_\mathbf{w} \frac{1}{2}||\mathbf{Xw - y}||^2_2 + \alpha||\mathbf{w}||_1
  $$
- No closed-form solution
- To find out the weight, use **Least angle regression**