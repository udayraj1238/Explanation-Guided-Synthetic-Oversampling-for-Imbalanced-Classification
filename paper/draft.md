# Attribution-Constrained SMOTE: Guiding Synthetic Oversampling with Feature Importance for Imbalanced Classification

## 1. Introduction

Learning from imbalanced datasets remains a foundational challenge in machine learning. When the majority class vastly outnumbers the minority class, standard classifiers naturally bias toward the majority, often producing decision boundaries that misclassify the rare, high-value instances that practitioners care most about. While algorithmic level approaches (like cost-sensitive learning) address this by modifying the training objective, data-level approaches—particularly synthetic oversampling—remain highly popular because they are classifier-agnostic and easy to integrate into existing pipelines.

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
