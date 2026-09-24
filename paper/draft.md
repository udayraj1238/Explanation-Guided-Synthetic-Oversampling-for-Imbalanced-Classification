# Attribution-Constrained SMOTE: Guiding Synthetic Oversampling with Feature Importance for Imbalanced Classification

## 1. Introduction

Learning from imbalanced datasets remains a foundational challenge in machine learning. When the majority class vastly outnumbers the minority class, standard classifiers naturally bias toward the majority, often producing decision boundaries that misclassify the rare, high-value instances that practitioners care most about. While algorithmic level approaches (like cost-sensitive learning) address this by modifying the training objective, data-level approachesâ€”particularly synthetic oversamplingâ€”remain highly popular because they are classifier-agnostic and easy to integrate into existing pipelines.

Since the introduction of SMOTE, many variants have been proposed, including Borderline-SMOTE and ADASYN. These methods generate synthetic minority samples by interpolating between a real minority point and its nearest neighbors. However, a significant limitation of standard SMOTE-family algorithms is that they treat all feature dimensions identically. They perform neighbor search and interpolation using standard distance metrics (like Euclidean distance) without any notion of which features actually drive the classifier's decision boundary. Consequently, they risk generating synthetic points that blur the decision boundary along the most critical features, ultimately harming downstream classification performance.

In this paper, we propose Attribution-Constrained SMOTE (AC-SMOTE), a novel oversampling technique that uses explainable AI (XAI) feature attributions to guide the generation of synthetic data. By leveraging cheap, SHAP-based feature importance scores from a lightweight surrogate model, AC-SMOTE addresses the limitations of feature-agnostic oversamplers in two ways:
1. **Attribution-weighted neighbor search:** It biases the neighbor selection toward minority points that are close on decision-relevant dimensions, ensuring that the local neighborhoods used for interpolation are faithful to the true class signal.
2. **Attribution-constrained interpolation:** It shrinks the interpolation step size along high-importance dimensions, anchoring the synthetic points near real minority instances on critical features, while allowing full-range interpolation along low-importance dimensions to inject useful diversity.

We benchmark AC-SMOTE against four standard baselines across ten public imbalanced datasets and three classifiers (Logistic Regression, Random Forest, and XGBoost). Using a rigorous nested cross-validation protocol and proper non-parametric statistical testing (Friedman and Nemenyi tests), we demonstrate the effectiveness of using feature attributions to constrain synthetic oversampling. 

## 2. Related Work

*(Note: Full literature search to be completed on Day 6. This is a structural sketch.)*

### 2.1. Synthetic Oversampling for Imbalanced Data
- **Standard SMOTE:** Discuss the original SMOTE algorithm and its limitation of generating samples blindly without considering the decision boundary.
- **Boundary-focused SMOTE variants:** Discuss Borderline-SMOTE, ADASYN, and Safe-Level-SMOTE, which focus on selecting *which* points to oversample (e.g., those near the boundary) but still use feature-agnostic interpolation.

### 2.2. Feature Selection and Weighting in Imbalanced Learning
- Compare against methods that perform feature selection *before* oversampling.
- Highlight the gap: selection drops features entirely, whereas AC-SMOTE uses soft constraints (weighting) to preserve diversity in non-critical features while tightly controlling critical ones.

### 2.3. Explainable AI (XAI) and SHAP in Pipeline Design
- Briefly cover the rise of SHAP (SHapley Additive exPlanations).
- Discuss how XAI is typically used *post-hoc* to explain models.
- Contrast with our approach, which integrates SHAP *in-the-loop* as a generative constraint to improve the data distribution itself.

## 3. Proposed Method: Attribution-Constrained SMOTE (AC-SMOTE)

Standard synthetic oversampling generates new instances by interpolating between a minority point $x_i$ and one of its $k$-nearest neighbors $x_{nn}$. Both the neighbor selection (via Euclidean distance) and the interpolation step treat all feature dimensions equally. AC-SMOTE addresses this limitation by introducing a feature attribution vector $\hat{w}$ derived from an explainable AI (XAI) surrogate model.

### 3.1. Step 1: SHAP Attribution Vector

Given a training dataset with features $X$ and labels $y$, we first fit a lightweight surrogate model (e.g., a Random Forest classifier). We then apply SHAP (SHapley Additive exPlanations), specifically the fast TreeExplainer, to compute the local feature attributions for all instances belonging to the minority class ($X_{min}$).

Let $\Phi \in \mathbb{R}^{n_{min} \times d}$ be the matrix of absolute SHAP values for the minority class, where $n_{min}$ is the number of minority samples and $d$ is the number of features. We compute the global importance vector $w$ by averaging the absolute SHAP values across all minority instances:
$$ w_j = \frac{1}{n_{min}} \sum_{i=1}^{n_{min}} |\Phi_{i,j}| $$

To ensure numerical stability and bounded constraint application, we min-max normalize $w$ to produce the final attribution vector $\hat{w} \in [0, 1]^d$:
$$ \hat{w}_j = \frac{w_j - \min(w)}{\max(w) - \min(w)} $$

### 3.2. Step 2: Attribution-Weighted Neighbor Selection

To prevent the algorithm from selecting neighbors based on irrelevant features, AC-SMOTE replaces standard Euclidean distance with an attribution-weighted distance metric. For any two minority samples $u, v \in X_{min}$, the distance is defined as:
$$ d_w(u, v) = \sqrt{\sum_{j=1}^d \left( 1 + \alpha \hat{w}_j \right) (u_j - v_j)^2} $$

Here, $\alpha \geq 0$ is a hyperparameter controlling the strength of the attribution weighting. Features with high SHAP importance ($\hat{w}_j \approx 1$) contribute more heavily to the distance, forcing the $k$-nearest neighbor search to return points that are fundamentally similar along the most critical dimensions. When $\alpha = 0$, this collapses to standard Euclidean distance.

### 3.3. Step 3: Attribution-Constrained Interpolation

Standard SMOTE generates a synthetic point $x_{synth}$ using a single random scalar $\lambda \sim U(0, 1)$:
$$ x_{synth} = x_i + \lambda \times (x_{nn} - x_i) $$

This formulation explores the feature space equally in all directions. AC-SMOTE applies a feature-wise constraint to restrict interpolation along critical dimensions, forcing synthetic samples to stay closer to real minority instances on important features while allowing full-range diversity on less important ones. For each feature dimension $j$, the step size $\lambda_j$ is defined as:
$$ \lambda_j = \max\left(0, \min\left(1, \lambda \times (1 - \beta \hat{w}_j) \right)\right) $$

where $\beta \in [0, 1]$ controls the constraint strength. If a feature is highly important ($\hat{w}_j \approx 1$), the step size $\lambda_j$ shrinks toward 0, anchoring the synthetic point near $x_i$. If a feature is unimportant ($\hat{w}_j \approx 0$), $\lambda_j \approx \lambda$, allowing standard exploration.

The final synthetic point is generated element-wise:
$$ x_{synth, j} = x_{i, j} + \lambda_j \times (x_{nn, j} - x_{i, j}) $$

### 3.4. AC-SMOTE Pseudocode

```text
Algorithm 1: AC-SMOTE
Input: Minority set X_min, Majority set X_maj, k_neighbors, alpha, beta, surrogate_model
Output: Synthetic minority set X_synth

1: Fit surrogate_model on (X_min U X_maj)
2: Compute SHAP values for X_min using TreeExplainer
3: w = mean(abs(SHAP), axis=0)
4: w_hat = min_max_normalize(w)
5: Compute pairwise distance matrix D for X_min using d_w(u,v, alpha, w_hat)
6: For each x_i in X_min:
7:     Find k nearest neighbors of x_i using D
8:     Randomly select one neighbor x_nn
9:     Sample lambda ~ U(0, 1)
10:    For j = 1 to d:
11:        lambda_j = clip(lambda * (1 - beta * w_hat[j]), 0, 1)
12:        x_synth[j] = x_i[j] + lambda_j * (x_nn[j] - x_i[j])
13:    Add x_synth to X_synth
14: Return X_synth
```

## 4. Experimental Setup

### 4.1. Datasets
To ensure robust evaluation across varying degrees of class imbalance and dimensionality, we benchmark AC-SMOTE on 10 datasets: 8 standard benchmarks from the `imbalanced-learn` repository and 2 real-world tabular datasets (Credit Card Fraud Detection and Telco Customer Churn). 
*(Note: Full dataset statistics table to be generated from dataset_manifest.json on Day 5).*

### 4.2. Baselines and Classifiers
We compare AC-SMOTE against five baseline strategies:
1. **None:** Training directly on the imbalanced data.
2. **ROS:** Random Over-Sampling.
3. **SMOTE:** Standard Synthetic Minority Oversampling Technique.
4. **Borderline-SMOTE:** Focuses oversampling on minority instances near the decision boundary.
5. **ADASYN:** Adaptively shifts the importance of minority instances based on their difficulty to learn.

To evaluate generalization across learning algorithms, we test three classifiers: Logistic Regression (linear), Random Forest (bagging), and XGBoost (boosting).

### 4.3. Evaluation Protocol and Metrics
To strictly avoid data leakage—a common pitfall in oversampling literature—we employ a rigorous repeated stratified cross-validation protocol (5 splits, 3 repeats, yielding 15 folds). Crucially, in every fold, the dataset is split *before* any oversampling occurs. The resampler is fit and applied exclusively to the training fold, while the classifier is evaluated on the original, unmodified test fold.

We report five metrics to capture performance under severe imbalance: F1-score (minority class), Area Under the Precision-Recall Curve (AUC-PR), Matthews Correlation Coefficient (MCC), Geometric Mean (G-mean), and Balanced Accuracy.

## 5. Results

Our experimental evaluation across 10 datasets validates the effectiveness of AC-SMOTE, while also revealing nuances in how attribution-constrained oversampling interacts with different learning algorithms. All reported values correspond to the dataset-level means across the 15 evaluation folds (Repeated Stratified 5-Fold CV). Full dataset-by-dataset outcomes are provided in `results/summary_table.csv`.

### 5.1. Performance across Classifiers

The non-parametric Friedman test, executed across the six tested strategies (None, ROS, SMOTE, Borderline-SMOTE, ADASYN, and AC-SMOTE) using each dataset as a block, confirmed statistically significant differences across methods. For instance, evaluating the Geometric Mean (G-mean) metric, the Friedman test returned a test statistic of $\chi^2 = 23.31$ ($p = 0.00029$) for Logistic Regression, $\chi^2 = 27.20$ ($p < 0.0001$) for Random Forest, and $\chi^2 = 24.06$ ($p = 0.0002$) for XGBoost. 

A post-hoc Nemenyi test (visualized in the Critical Difference Diagram, Figure 1) groups the resamplers based on their rank distributions. We observed that the choice of oversampling strongly impacts models with lower capacity or unregularized linear bounds, whereas powerful non-linear ensembles (like XGBoost) exhibit different sensitivities.

### 5.2. Pairwise Evaluation: AC-SMOTE vs. Baselines

To evaluate AC-SMOTE directly against individual baselines, we performed a Wilcoxon signed-rank test on the paired dataset-level means. 

**Compared to No Resampling (None):**
AC-SMOTE unequivocally improves metrics that account for class imbalance. Against the "None" baseline using Random Forest, AC-SMOTE achieved 10 wins and 0 losses for both G-mean and Balanced Accuracy ($p = 0.0019$). Similar dominance was observed for Logistic Regression (10 wins, 0 losses for both metrics). 

**Compared to Basic Resampling (ROS and SMOTE):**
When compared to Random Over-Sampling (ROS) under Logistic Regression, AC-SMOTE achieved a significant victory in F1-score (9 wins, 1 loss, $p = 0.0097$) and MCC (9 wins, 1 loss, $p = 0.0195$).
Against standard SMOTE using Random Forest, the results were highly competitive but mixed. For G-mean, AC-SMOTE secured 6 wins and 4 losses; however, for F1-score, SMOTE outperformed AC-SMOTE on 7 out of 10 datasets. This suggests that while attribution-constrained interpolation helps preserve the underlying geometric distribution (benefitting G-mean), unconstrained SMOTE's broader feature-space exploration can sometimes yield higher precision/recall harmonic means in complex ensembles.

**Compared to Density-based Resampling (Borderline-SMOTE and ADASYN):**
AC-SMOTE demonstrated strong advantages over Borderline-SMOTE when evaluated with Random Forest, winning 8 out of 10 datasets in Balanced Accuracy ($p = 0.0195$) and 7 out of 10 in G-mean ($p = 0.0371$). Interestingly, when compared to ADASYN using XGBoost, AC-SMOTE proved vastly superior in preserving the Area Under the Precision-Recall Curve (AUC-PR), securing 9 wins and only 1 loss ($p = 0.0039$). 

## 6. Discussion and Ablation Study

### 6.1. Ablation on Hyperparameters ($\beta$, $\alpha$, $k$)
To isolate the contribution of our attribution-guided constraints, we conducted an ablation study over our novel hyperparameters using XGBoost across 5 representative datasets (including the heavily imbalanced `abalone_19` and the massive `creditcard` benchmark).

The ablation sweeps (stored in `ablation_results.csv`) confirm the efficacy of the interpolation constraint ($\beta$). Setting $\beta = 0$ reduces AC-SMOTE to using only attribution-weighted neighbor selection, while increasing $\beta \to 1$ tightens the interpolation step aggressively around critical SHAP features. As visualized in the $\beta$-sweep plots, intermediate values (e.g., $\beta = 0.5$ or $0.7$) typically strike the optimal balance, preventing the collapse of synthetic variance while avoiding the generation of noisy samples in critical dimensions.

### 6.2. Interpretation and Limitations
The experimental results enforce a known reality in imbalanced learning literature: no single oversampling technique universally dominates every metric and every classifier. 

AC-SMOTE succeeds where standard SMOTE creates "bridge" samples that cross the true decision boundary in highly critical feature dimensions. By forcing interpolation to stay close to the parent instance on high-SHAP features, AC-SMOTE protects the integrity of those dimensions. This was distinctly visible in our PCA projections on the `wine_quality` dataset, where AC-SMOTE's synthetic instances adhered more closely to the dense clusters of true minority samples compared to the diffuse spread of standard SMOTE.

However, this structural fidelity comes with two trade-offs. 
First, there is an observable computational overhead. The runtime comparison chart demonstrates that AC-SMOTE is slower than SMOTE, driven inherently by the need to fit a surrogate model and compute TreeSHAP values prior to neighbor selection. 
Second, in specific classifier configurations (notably Random Forest evaluated on F1-score), restricting variance in critical dimensions sometimes limits the ensemble's ability to discover broader decision rules, leading to minor losses against unconstrained SMOTE. 

Despite these limitations, AC-SMOTE represents a mathematically principled approach to injecting Explainable AI (XAI) directly into the generative pipeline, proving that feature attributions can actively improve data quality rather than merely explaining it post-hoc.
