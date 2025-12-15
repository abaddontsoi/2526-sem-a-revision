# Machine Learning & Computer Vision: Comprehensive Notes

## Part 1: Classical Classification Algorithms

### K-Nearest Neighbors (KNN)
- **Type:** Non-parametric, Instance-based (Lazy Learning).
- **Pros**
  - Simple implementation; no training phase (just storage).
  - Naturally handles non-linear decision boundaries.
  - Effective for multi-class problems without modification.
- **Cons**
  - **Slow Inference:** Computationally expensive at prediction time (calculates distance to all training points).
  - **Memory Intensive:** Must store the entire training dataset.
  - **Curse of Dimensionality:** Distance metrics lose meaning in very high dimensions.
  - Sensitive to feature scaling and irrelevant features.

### Naive Bayes (NB)
- **Type:** Generative Model ($P(x|y)$).
- **Pros**
  - Extremely fast training and prediction.
  - Performs well with high-dimensional data (e.g., text/spam).
  - Robust to irrelevant features.
  - Handles missing data naturally.
- **Cons**
  - **Independence Assumption:** Assumes features are independent given the class (rarely true in real life).
  - **Zero Frequency Problem:** Requires smoothing (e.g., Laplace) for unseen features.
  - Probability outputs are often poorly calibrated (too extreme), even if classification accuracy is good.

### Linear Discriminant Analysis (LDA)
- **Type:** Generative Model.
- **Pros**
  - **Supervised Dimensionality Reduction:** Projects data to maximize class separability (max dimensions = $C-1$).
  - Optimal classifier if data is Gaussian with equal covariance matrices.
  - Computationally efficient (closed-form solution).
- **Cons**
  - **Strict Assumptions:** Requires Gaussian distribution and shared covariance matrix across classes.
  - Sensitive to outliers.
  - Limited to linear decision boundaries.

### Logistic Regression (LR)
- **Type:** Discriminative Model ($P(y|x)$).
- **Pros**
  - Outputs well-calibrated probabilities.
  - Highly interpretable weights (feature importance).
  - Easy to update with new data (using SGD).
- **Cons**
  - Inherently a linear classifier (unless feature engineering is used).
  - Can overfit on high-dimensional data without regularization.
  - Sensitive to outliers.

### Support Vector Machines (SVM)
- **Type:** Discriminative (Max Margin).
- **Pros**
  - **Max Margin:** Theoretically robust to overfitting.
  - Effective in high-dimensional spaces (even when $d > n$).
  - Memory efficient (defined only by support vectors).
- **Cons**
  - No direct probability estimates (requires Platt scaling).
  - Sensitive to noise and parameter tuning ($C$).
  - **Kernel SVM:** Computationally expensive ($O(n^2)$ to $O(n^3)$) on large datasets.

---

## Part 2: Comparisons (The "Vs." Sections)

### Naive Bayes vs. LDA
| Feature | Naive Bayes | LDA |
| :--- | :--- | :--- |
| **Assumption** | Features are independent (Diagonal Covariance). | Features share correlations (Full Shared Covariance). |
| **Data Efficiency** | Better with small data. | Needs more data to estimate covariance matrix. |
| **Boundary** | Can be non-linear (Quadratic). | Strictly Linear. |

### Logistic Regression vs. Generative Models (NB/LDA)
| Feature | Logistic Regression | NB / LDA |
| :--- | :--- | :--- |
| **Type** | Discriminative (models $P(y\|x)$). | Generative (models $P(x\|y)$ and $P(y)$). |
| **Assumptions** | Few assumptions about data distribution. | Strong assumptions (Gaussian, Independence). |
| **Performance** | Generally higher accuracy with sufficient data. | Better with missing data or very small datasets. |

---

## Part 3: Regression & Optimization

### Linear Regression Variants
1.  **Standard Linear Regression:** Minimizes Sum of Squared Errors (SSE). Unstable with correlated features.
2.  **Ridge Regression (L2):** Adds $\lambda ||w||_2^2$. Shrinks coefficients towards zero but keeps them all. Solves multicollinearity.
3.  **LASSO Regression (L1):** Adds $\lambda ||w||_1$. Induces **sparsity** (sets some weights to exactly zero). Acts as feature selection.

### Gradient Descent (GD) Variants
| Variant | Batch Size | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Batch GD** | All Data | Stable convergence; exact gradient. | Slow; high memory usage; can get stuck in saddle points. |
| **Stochastic GD** | 1 Sample | Fast; escapes local minima; low memory. | Noisy convergence; requires learning rate decay. |
| **Mini-Batch GD** | $N$ Samples | Best of both worlds; utilizes GPU vectorization. | Requires tuning batch size. |

### RANSAC (Random Sample Consensus)
- **Purpose:** Robust fitting of models (e.g., lines, homographies) in the presence of many outliers.
- **Pros:** extremely robust to outliers.
- **Cons:** Non-deterministic; requires a threshold parameter; no guarantee of finding the optimal solution if iterations are too low.

---

## Part 4: Unsupervised Learning & Dimensionality Reduction

### Clustering
- **K-Means:** Hard assignment. Assumes spherical clusters of similar size. Fast but sensitive to initialization and outliers.
- **Gaussian Mixture Models (GMM):** Soft assignment (probabilistic). Assumes elliptical clusters. Uses Expectation-Maximization (EM). More flexible but slower.

### Dimensionality Reduction
- **PCA (Principal Component Analysis):**
  - **Unsupervised.**
  - Finds orthogonal directions of maximum variance.
  - **Cons:** Sensitive to scaling; assumes linear correlations; "Variance" does not always equal "Information."
- **SVD (Singular Value Decomposition):**
  - The mathematical matrix factorization technique that underpins PCA.
  - More numerically stable than eigendecomposition.

---

## Part 5: Deep Learning

### Architectures
- **MLP (Multi-Layer Perceptron):** Dense connections. Good for tabular data. Prone to overfitting on images.
- **CNN (Convolutional Neural Network):**
  - **Inductive Bias:** Translation invariance and locality.
  - **Pros:** Parameter sharing (efficient), learns hierarchical features.

### Regularization & Training
- **Dropout:** Randomly deactivates neurons during training. Prevents co-adaptation of features (ensemble effect).
- **Early Stopping:** Stops training when validation error rises. Prevents overfitting.
- **Data Augmentation:** Artificially expands dataset (flip, rotate, crop). Crucial for generalization in computer vision.
- **Batch Normalization/Whitening:** Stabilizes learning by normalizing layer inputs (zero mean, unit variance). Allows higher learning rates.

---

## Part 6: Computer Vision & Image Processing

### Tasks
- **Object Detection:** Bounding box + Class label (e.g., YOLO, Faster R-CNN).
- **Semantic Segmentation:** Pixel-level classification (all "cars" are the same color).
- **Instance Segmentation:** Pixel-level classification distinguishing individual objects (Car A vs. Car B).

### Image Restoration
- **Denoising:** Removing noise. Deep learning (DnCNN) outperforms filters but is slower.
- **Super-Resolution:** Upscaling images. GANs often used to hallucinate high-frequency details.

### Quality Metrics
- **MSE/PSNR:**
  - **Pros:** Simple, differentiable math.
  - **Cons:** Poor correlation with human perception (penalizes slight shifts heavily).
- **SSIM (Structural Similarity):**
  - **Pros:** Captures luminance, contrast, and structure. Closer to human vision.
- **LPIPS (Learned Perceptual Image Patch Similarity):**
  - **Pros:** Uses deep network features to measure similarity. Best match for human perception.
  - **Cons:** Computationally heavy.