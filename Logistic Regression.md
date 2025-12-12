## Logistic Regression
- Use sigmoid function
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
- Only gives binary output
    - $p(y=1|\mathbf{x}) = \sigma(f(\mathbf{x}))$
    - $p(y=-1|\mathbf{x}) = 1 - \sigma(f(\mathbf{x})) = \sigma(-f(\mathbf{x}))$
    - Unified, $p(y|\mathbf{x}) = \sigma(yf(\mathbf{x}))$
- With a function
$$
z = f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b
$$
- Target is to maximize log likelihood with additional $p(\mathbf{w}) \propto \exp (-\frac{1}{C}\mathbf{w}^T\mathbf{w})$ to prevent overfitting
$$
(\mathbf{w}^*, b^*) = \arg \min \frac{1}{C}\mathbf{w}^T\mathbf{w}+\frac{1}{M} \sum_i \log (1 + \exp(-y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b)))
$$
- The loss function $L(z) = \log (1 + \exp(-z))$ with $z = y^{(i)}f(\mathbf{x}^{(i)})$
    - Gives $\gt 1$ when $z < 0$, which is missclassified
    - Gives $\lt 1$ when $z > 0$, which is correctly classified
    - Gives $= 1$ when $z = 0$, which is on the decision boundary
- The minimization does not have closed form solution
    - Use gradient descent
    $$
        \mathbf{w} \leftarrow \mathbf{w} - \eta\frac{\partial f}{\partial \mathbf{w}}
    $$
    
## Application to multiclass classification with LR
- Instead of pair-wise LR classifiers
    - Use 1-vs-rest classifiers
    - In total N classifiers needed
- Or define probabilities with **softmax** function
$$
p(y=c|\mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x})}{\sum_i \exp(\mathbf{w}_i^T \mathbf{x})}
$$
    - Then estimate $\mathbf{w}$ with MLE