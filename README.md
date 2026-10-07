# Breast Cancer Diagnosis: Unsupervised and Supervised Learning

Machine Learning I final project, TECNUN Universidad de Navarra, Spring 2026.

Analysis of the Wisconsin Breast Cancer (Diagnostic) dataset from two angles: unsupervised learning to find structure in the data without labels, and supervised learning to predict whether a tumor is malignant or benign. The two parts are then connected to test whether the clusters match the true diagnosis and whether they explain classifier errors.

## Repository contents

| File | Description |
|---|---|
| `Final_project.ipynb` | Full analysis notebook (EDA, preprocessing, unsupervised, supervised, connection analysis) |
| `final_report_kinga_kerekes.pdf` | Written project report |
| `requirements.txt` | Python dependencies |

## Dataset

- Wisconsin Breast Cancer (Diagnostic), loaded from scikit-learn
- 569 samples, 30 numerical features (mean, standard error and worst value of 10 nuclear measurements from digitized FNA images)
- Classes: 357 benign (62.7%), 212 malignant (37.3%); malignant is the positive class
- No missing values

Source: Wolberg, Mangasarian, Street & Street (1993), UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B

## Methods

**Preprocessing**
- Stratified 80/20 train/test split
- `StandardScaler` fitted on training data only (no leakage)

**Unsupervised**
- PCA (7 components capture 90% of variance, 10 capture 95%)
- t-SNE (perplexity selected by sensitivity analysis)
- K-Means, Gaussian Mixture Models, Hierarchical clustering (Ward linkage)
- Evaluation: silhouette score; post-hoc verification with ARI, recall and F1

**Supervised**
- Logistic Regression, Linear SVM, k-Nearest Neighbors, Random Forest
- Hyperparameters tuned with 5-fold stratified cross-validation, recall as scoring metric
- Cost-sensitive variants (`class_weight='balanced'`)
- Primary metric: recall on the malignant class; secondary metric: F1

**Connection analysis**
- Effect of PCA on prediction
- Agreement between clusters and true labels
- Cluster membership as an additional feature
- Location of misclassified samples relative to GMM uncertainty

## Key results

### Unsupervised (30D baseline silhouette)

| Method | Silhouette | ARI (post-hoc) | Recall (post-hoc) |
|---|---|---|---|
| K-Means | 0.3432 | 0.6622 | 0.8412 |
| GMM | 0.3142 | 0.7524 | 0.9294 |
| Hierarchical (Ward) | 0.2886 | 0.6212 | 0.9235 |

### Supervised (test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 1.0000 | 0.9524 | 0.9756 | 0.9980 |
| Linear SVM | 0.9386 | 0.9487 | 0.8810 | 0.9136 | 0.9679 |
| kNN (k=3) | 0.9386 | 0.9730 | 0.8571 | 0.9114 | 0.9825 |
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9929 |
| Logistic Regression (balanced) | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9977 |
| Linear SVM (balanced) | 0.9912 | 1.0000 | 0.9762 | 0.9880 | 0.9983 |
| Random Forest (balanced) | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9965 |

### Main findings

- The data contain genuine cluster structure: GMM recovers 93% of malignant cases without using labels.
- Balanced Linear SVM is the strongest model (recall 0.9762, F1 0.9880). Class weighting removed its overfitting (best C dropped from 29.76 to 0.002).
- Balanced Logistic Regression is the recommended model for clinical use: interpretable coefficients, calibrated probabilities, negligible train-test gap.
- A 2-component PCA projection matches the recall of all 30 features for balanced Logistic Regression.
- Adding K-Means cluster membership as a feature did not improve prediction.
- Misclassified samples concentrate in the overlap zone along PC1 that GMM identifies as ambiguous.

## How to run

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Final_project.ipynb
```

## Limitations

- Evaluation uses a single held-out split of one dataset; no external validation.
- The dataset is a clean benchmark, so results are an upper bound on real-world performance.
- Features come from an image-processing pipeline whose errors are not visible in the analysis.
- With 569 samples, estimates are sensitive to the specific split.

## Possible improvements

- Kernel SVM and gradient boosting (e.g. XGBoost)
- Decision threshold optimization via precision-recall curves

## Author

Kinga Kerekes

## References

- Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis. *Electronic Imaging*.
- Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository.
