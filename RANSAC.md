## RANSAC
- Random sample consensus
- Repeat the followings
  1. Learn a model through a minimum subset of dataset
  2. Use remaining data to check whether it is in inliner threshold, i.e. not too far away from prediction
  3. A consensus set is formed by inliners
  4. Save the model with the largest consensus set
- Requires $T$ iterations
  $$
  \begin{aligned}
  (1 - (1-e)^s)^T \lt 1 - \delta \\
  T \gt \frac{\log(1-\delta)}{\log(1 - (1-e)^s)}
  \end{aligned}
  $$
  where $e$ is outliners, $s$ is number of points to fit a model, $\delta$ is success rate
- Can be interpreted as
  - Probability of $T$ iterations with at least 1 outliner is less than the possibility of failure
- $T \propto s$ steeply