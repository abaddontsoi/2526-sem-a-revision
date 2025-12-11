## Bayes Optimal Classifier
- Generative model
- Probabilistic
- Use a probility distribution, e.g. Gaussian, Poison, ...
- Find out the parameter for the distribution with log-MLE
$$
(\hat{\mu_c}, \hat{\sigma_c}^2) = {\arg \max}_{\hat{\mu_c}, \hat{\sigma_c}^2} \sum \log p(x^{(i)} | y^{(i)}; \mu_c, \sigma_c^2)
$$
- The solution is sample mean and sample variance
- To perform classification, need to find out $p(y|\mathbf{x})$ with $p(y)$ and $p(\mathbf{x}|y)$ using Bayes' Rule
$$
p(y|\mathbf{x}) = \frac{p(\mathbf{x|y}) p(y)}{\sum_i p(\mathbf{x}|y=i)}
$$