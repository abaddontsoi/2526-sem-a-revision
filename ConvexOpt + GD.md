## Convex set
- Any point in any **line segment** is in the set
$$
\alpha\mathbf{x}^{(1)} + (1 - \alpha)\mathbf{x}^{(2)} \in \mathcal{X}
$$

## Hyperplane and halftspace
- In the form $\{\mathbf{x} | \mathbf{a}^T\mathbf{x} = b\}$
- In the form $\{\mathbf{x} | \mathbf{a}^T\mathbf{x} \le b\}$
- $\mathbf{a} \ne 0$
- Both convex

## Convex function
- Satisfies
$$
f(\alpha\mathbf{x}^{(1)} + (1 - \alpha)\mathbf{x}^{(2)}) \le \alpha f(\mathbf{x}^{(1)}) + (1 - \alpha) f(\mathbf{x^{(2)}}))
$$
- if $f$ is concave, then $-f$ is convex

## Gradient Descent
- Normal gradient descent, with all datapoints
$$
    \mathbf{w} \leftarrow \mathbf{w} - \eta\frac{\partial f}{\partial \mathbf{w}}
$$
- Stochastic GD
    - Similar to normal one, but use only 1 datapoint at a time
    - Expectation is same
    - Pros: Efficiency on time and space complexity, get out from bad local minima using its oscillations
    - Cons: Noisy step, take longer to achieve convergence as each step may not decrease, cannot be vectorized. 
- Mini-batch GD
    - Randomly sample a sub-dataset to perform normal GD, the obtained gradient is averaged