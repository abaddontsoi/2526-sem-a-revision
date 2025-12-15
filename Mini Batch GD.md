# Mini-batch Gradient Descent

## Introduction
Hi, beginner! Mini-batch GD is a smart way to update model params—use small groups of data instead of all or one. Balance speed and stability.

From lecture: Compromise between full GD and SGD.

## How It Works
- At iteration t: Sample mini-batch B subset D (training data).
- Gradient: ∇ℓ(B; θ^{(t)}) = (1/|B|) ∑_{(x,y) in B} ∇ℓ(x,y; θ^{(t)}).
- Update: θ^{(t+1)} = θ^{(t)} - α ∇ℓ(B; θ^{(t)}).
- Repeat till convergence.

Batch size: e.g., 32-256 in deep learning.

## Advantages
- Faster than full GD (less compute per update).
- Less noisy than SGD (averages batch).
- Popular in deep learning; vectorizes well.
- Good for large datasets.

## Disadvantages
- Needs tuning (batch size, α).
- Can stuck in local mins.
- Variance if batch small.

## Examples
Used in training neural nets, logistic regression.

Vs Full GD: Mini-batch faster, scalable.
Vs SGD: Mini-batch smoother convergence.
