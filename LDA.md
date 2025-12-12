## Linear Disriminat Analysis
- Based on Naive Bayes
    - But not assumes independent features
- Use multivariate Gaussian
$$
p(\mathbf{x} | y=c) = \frac{1}{|(2\pi)^N \Sigma|^{1/2}} \exp(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu_c}) ^T \Sigma^{-1}(\mathbf{x} - \mathbf{\mu_c}))
$$
- By solving the MLE, the required parameters of multivariate Gaussian are:
    - Class probability
    $$
    p(y = c) = \frac{M_c}{M}
    $$
    - Class mean
    $$
    \frac{\sum_i \mathbf{x}^{(i)}}{M_c}
    $$
    - Shared covariance
    $$
    \Sigma = \frac{1}{M} \sum_i (\mathbf{x}^{(i)} - \mathbf{\mu}_{y^{(i)}})(\mathbf{x}^{(i)} - \mathbf{\mu}_{y^{(i)}})^T
    $$ 
    where $\mathbf{\mu}_{y^{(i)}}$ is class mean
- Decision boundary is linear in $\mathbf{x}$