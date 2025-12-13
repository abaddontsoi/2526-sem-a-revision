## Sequential Minimal Optimization
- Choose 2 dual variable (Lagrangian multiplier) to optimize
  - Other multipliers are fixed
- Dual var is choosed when violating KKT condition
- In SVM, typically choose 2 $\alpha$'s to optimize
  - Choose first with violating most of the KKT
  - Choose second one with greatest expected change
    - With $|E_i - E_j|$ large, while $E_k$ is prediction error in sample $k$