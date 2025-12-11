## Normal KNN
- Class is determined by majority votes of K nearest neighbor
$$
f_{\text{KNN}} = {\arg \max}_{c \in \{1, ... C\}} \sum I[y^{(i)} = c]
$$

## Weighted KNN
- Applies a weight to each vote
- Classifying function is similar to Normal KNN, but with extra weight for indicator function
- Final classifier is divided by sum of all weights $w_i$
- $w_i = \exp(-\alpha d)$

## Pros and Cons
- Pros
    - No training (lazy learning)
    - Converges correct dicision surface as dataset grows
- Cons
    - High time and space complexity for storing and searching as features and dataset grows