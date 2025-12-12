## Naive Bayes
- Deal with high dimension featuers, e.g. $\mathbf{x} \in \mathbb{R}^N$
- Assume features are independent
- Features can have different distribution, e.g. Gaussian, Poison, ...
- Classification function:
$$
f_{NB}(\mathbf{x}) = \arg \max_c p(y=c) \prod_j p(x_j | y=c; \theta)
$$
where $\theta$ is parameter of a feature's ditritubtion
- Target is to learn the parameter $\theta$
    - i.e. Where to place the classifier