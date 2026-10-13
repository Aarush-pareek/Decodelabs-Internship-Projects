# Project Report: AI Data Classification System

**Internship Project 2 - Data Classification Using AI**

---

## 1. Introduction

Classification is one of the most common tasks in machine learning: given measurements describing something, decide which category it belongs to. Email filters decide spam or not spam, doctors' tools flag suspicious scans, and a botanist's tool can name a flower from its measurements.

This project builds a complete, small classification system around that idea. It is deliberately simple in its algorithms and careful in its engineering: every stage of the supervised-learning pipeline is implemented in its own module, tested, and shown in an interactive Streamlit dashboard.

## 2. Problem Statement

Given the measurements of an iris flower (sepal length, sepal width, petal length, petal width), predict its species (setosa, versicolor or virginica). More generally: given a labelled table of numeric features, train a model that classifies new, unseen samples, and measure honestly how well it does so.

## 3. Objective

The internship brief asks for a basic classification model built on a small dataset. This project therefore:

1. loads and understands a dataset,
2. splits the data into training and testing sets,
3. applies a simple classification algorithm,
4. trains and evaluates the model with appropriate metrics,
5. demonstrates supervised-learning fundamentals,
6. packages the result as a professional, demonstrable application.

## 4. Dataset

The **Iris** dataset (Ronald Fisher, 1936) is bundled with scikit-learn and also provided as `data/iris.csv`.

| Property | Value |
|---|---|
| Samples | 150 (50 per class, perfectly balanced) |
| Features | 4 numeric measurements in cm |
| Target | species: setosa, versicolor, virginica |
| Missing values | none |
| Exact duplicate rows | 1 |

**Understanding the data.** The dashboard (Section 2) and `notebooks/exploration.ipynb` show the following:

- Setosa is clearly separated from the other two species, especially by petal length and petal width.
- Versicolor and virginica overlap, so a few mistakes between them are expected.
- Petal length and petal width are strongly correlated (r about 0.96). They carry similar information.
- The dataset is balanced, so accuracy is a fair headline metric here. On imbalanced data it would not be (see Section 10).

**Validation.** The loader never assumes the data is perfect. It rejects empty files, invalid CSVs, a missing target column, datasets without numeric features, fewer than 20 labelled rows, a target with a single class, a class with fewer than two samples, and targets with more than eight distinct values (probably a continuous or ID column). Each rejection produces a plain-language message in the interface.

## 5. Data Preprocessing

Preprocessing is split into two kinds of work, for a reason explained below.

**Row-level cleaning** (`src/preprocessing.py`, `clean_data`):

- rows without a label are removed, because there is nothing to learn from them;
- exact duplicate rows are removed. Iris contains one, and leaving duplicates in could place the same sample in both the training and test sets, inflating the test score.

Cleaning 150 rows leaves 149.

**Learned transformations** (inside the model pipeline, `src/model_training.py`):

- **Median imputation** fills any missing feature values with the median of the *training* data.
- **Standardisation** (`StandardScaler`) rescales every feature to mean 0 and standard deviation 1.

**Why the second kind lives in the pipeline.** Imputation medians and scaling statistics are *learned from data*. If they were computed on the whole dataset before splitting, information about the test set would leak into training (data leakage) and the evaluation would be too optimistic. By making them steps of a scikit-learn `Pipeline`, they are fitted on the training rows only and applied unchanged to test and new data. A unit test confirms the fitted scaler's means equal the training-set means and not the full-dataset means.

## 6. Feature Selection

Feature and target selection (`select_features_and_target`) checks that the chosen columns exist, are numeric, are not empty, and that the target is not also used as a feature. In the dashboard the user can deselect features in Section 3. For Iris all four features are used, because they are few, cheap, and each adds information.

The report also keeps the target separate from the features at every step: the model only ever receives `X` (features), and `y` (labels) is used for learning and scoring.

## 7. Train/Test Split

Supervised learning needs data to learn from and *different* data to be judged on. A model scored on the data it learned from can simply memorise it.

The split (`split_data`) uses scikit-learn's `train_test_split` with:

- **test size 20% by default** (adjustable from 10% to 50%): 119 training rows and 30 test rows after cleaning;
- **shuffling** to remove any ordering bias (the raw Iris file is sorted by species);
- **stratification**, so each class keeps the same proportion in both sets (10 test samples per class);
- **a fixed `random_state` of 42**, so every run gives the same split and the same results.

The test set is used exactly once, to report final performance. Choosing between models is done with 5-fold cross-validation inside the training set.

## 8. Machine Learning Algorithm

**Logistic Regression** is the primary model.

- For each class it learns a weight per feature. A sample's weighted sum (its "score") for each class is converted by the softmax function into probabilities that add up to 1. The class with the highest probability is the prediction.
- Reasons for choosing it: it is a simple, standard first classifier, trains in milliseconds, gives class probabilities (the prediction confidence), and its weights are interpretable, which supports the explanation shown with each prediction.
- It needs scaled features to train reliably, which the pipeline provides.

Two further algorithms are included for comparison:

- **Decision Tree** (maximum depth 4): learns if/else questions about feature values. The depth limit prevents it from memorising the training data.
- **K-Nearest Neighbors** (K = 5): classifies a sample by a majority vote among its five closest training samples. It relies on distances, so scaling is essential. The dashboard includes a cross-validated chart for choosing K; on this dataset K = 5 gives the lowest error (tied with K = 3).

The reference slides for this project illustrate the workflow with K-Nearest Neighbors; this project defaults to Logistic Regression and keeps KNN (K = 5) selectable. Changing the default is a one-line edit of `DEFAULT_MODEL` in `src/config.py`.

## 9. Model Training

Training (`train_pipeline`) follows the standard scikit-learn pattern the reference slides describe:

```python
pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
pipeline.fit(X_train, y_train)      # learn
predictions = pipeline.predict(X_test)   # apply
```

Training is wrapped in error handling, so a failure produces a clear message and not a traceback. After a successful run the fitted pipeline and its metadata (feature names, class names, training-set statistics, evaluation results) are saved together as `models/trained_model.pkl` with joblib. Because preprocessing is inside the saved pipeline, a reloaded model transforms new data exactly as it did during training.

## 10. Evaluation Metrics

All metrics are computed on the held-out test set.

| Metric | Plain-language meaning |
|---|---|
| Accuracy | Share of all predictions that were correct |
| Precision | Of the samples predicted as a class, how many really were that class |
| Recall | Of the samples that truly belong to a class, how many were found |
| F1 score | Harmonic mean of precision and recall; low if either is low |
| Confusion matrix | Table of true class (rows) against predicted class (columns); the diagonal is correct |

Precision, recall and F1 are **macro-averaged** (computed per class, then averaged) so that every class counts equally.

**Why more than accuracy.** If 99% of samples belong to one class, a model that always predicts that class scores 99% accuracy while being useless for the rare class. The dashboard therefore (a) shows per-class precision, recall and F1, (b) warns when the smallest class is less than half the size of the largest, and (c) a test in the project demonstrates the effect on synthetic imbalanced data.

**Overfitting check.** The dashboard compares training accuracy with testing accuracy. A gap above 10 percentage points raises a warning.

## 11. Results

Configuration: Logistic Regression, 80/20 stratified split, seed 42.

| Measure | Value |
|---|---|
| Training accuracy | 95.8% |
| Testing accuracy | 93.3% (28 / 30) |
| Precision / Recall / F1 (macro) | 93.3% / 93.3% / 93.3% |
| 5-fold CV accuracy (training set) | 96.6% +/- 5.1% |

Confusion matrix (rows = true, columns = predicted):

| | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | 10 | 0 | 0 |
| **versicolor** | 0 | 9 | 1 |
| **virginica** | 0 | 1 | 9 |

Setosa is classified perfectly. The two errors are the expected versicolor/virginica confusion.

Comparison of algorithms on the same split:

| Model | CV accuracy (train) | Test accuracy | Test F1 (macro) |
|---|---|---|---|
| Logistic Regression | 96.6% | 93.3% | 93.3% |
| Decision Tree | 93.2% | 93.3% | 93.3% |
| K-Nearest Neighbors | 95.7% | 93.3% | 93.3% |

**Interpreting the numbers honestly.** With only 30 test samples, each mistake changes accuracy by 3.3 points, so a single split is a noisy estimate. Re-running the experiment on 50 different random splits gave:

| Model | Mean test accuracy | Std. dev. | Min | Max |
|---|---|---|---|---|
| Logistic Regression | 95.3% | 3.4% | 86.7% | 100% |
| Decision Tree | 94.6% | 3.1% | 86.7% | 100% |
| K-Nearest Neighbors | 94.7% | 3.3% | 86.7% | 100% |

The honest conclusion is that all three algorithms reach roughly 94-95% on Iris and the differences between them are smaller than the split-to-split variation. The seed-42 split (93.3%) is slightly below average; the seed was fixed in advance and not tuned to look better.

## 12. Prediction System

Section 6 of the dashboard lets a user type the four measurements and press **Predict**.

1. **Input validation** (`validate_input`): all features must be present, numeric and finite. Invalid input produces a message naming the problem field.
2. **Range check**: a value outside the range seen in training triggers a warning, because the model is extrapolating.
3. **Prediction**: the saved pipeline predicts the class and its probabilities. The reported *confidence* is the probability of the predicted class.
4. **Explanation** ("Why this prediction?"):
   - *Distance (std)*: how far each of the user's values is from the average of the predicted class, in standard deviations. Small means typical.
   - *Pull toward class* (Logistic Regression): each standardised feature value multiplied by the model weight for the predicted class, showing which features pushed the decision.
   - A one-sentence summary names the most influential feature.

Example: the sample (6.3, 2.9, 5.6, 1.8) is predicted as **virginica** with 81.5% confidence, with petal length as the strongest driver; the remaining 18.5% goes to versicolor, reflecting the real overlap between those two species.

Confidence values are the model's own probability estimates; they are not guaranteed to be calibrated.

## 13. Challenges

1. **Small test set.** One error moves accuracy by 3.3 points. This was addressed by adding cross-validation and by measuring the spread over 50 splits so the reported result is not over-interpreted.
2. **Data leakage.** Scaling or imputing before splitting is a classic silent error. The solution was to put every learned transformation inside the pipeline and to add a test that proves the scaler is fitted on training data only.
3. **A hidden duplicate.** Iris contains one exact duplicate row. Handling it explicitly in cleaning avoids the same sample appearing on both sides of the split.
4. **Fair model comparison.** Choosing the "best" model on the test set would leak test information into the choice. Models are therefore compared with cross-validation on the training set, and the test score is reported alongside.
5. **Explaining a binary model correctly.** For two classes scikit-learn stores one weight row for the second class only. An early version of the explanation had the wrong sign for the first class; a dedicated test caught and now guards against it.
6. **Validation order.** A continuous target column first produced a confusing "class has fewer than 2 samples" message. Validation was reordered so users are told the real problem (too many distinct values).
7. **Friendly failure.** Real users upload empty, binary or non-numeric files. Every layer raises a specific exception that the interface converts into a clear message; an unexpected exception is logged and replaced with a generic message, so no traceback is shown during normal use.

**Limitations.** Numeric features only; a single hold-out split rather than repeated evaluation; fixed hyper-parameters; the Iris dataset is easy, so it demonstrates the pipeline rather than model-selection skill.

## 14. Future Improvements

- Hyper-parameter search (`GridSearchCV`) for regularisation strength, tree depth and K.
- Repeated stratified k-fold evaluation displayed in the dashboard.
- Categorical features (one-hot encoding in the pipeline) and text columns.
- ROC curves and calibration plots; threshold tuning for binary problems.
- Batch prediction from an uploaded CSV with downloadable results.
- Experiment tracking to compare several training runs.
- Docker image and cloud deployment.
- Progression toward the next stage in the programme: deep learning on images.

## 15. Conclusion

The project delivers the full supervised-learning loop: understand the data, clean it, split it fairly, train a simple classifier inside a leak-free pipeline, evaluate it with several metrics, and apply it to new data with an explanation. Logistic Regression reaches 93.3% test accuracy on the default split (about 95% on average across random splits), and two other algorithms, including the K-Nearest Neighbors (K = 5) workflow from the reference material, perform equivalently on this dataset.

Beyond the numbers, the main learning outcomes are practical: evaluation is only meaningful if the test set stays untouched; accuracy is not enough on its own; one split is a noisy estimate; and clean modular code with tests makes a model trustworthy and explainable.

---

## Internship Requirement Mapping

| Internship Requirement | Implementation |
|---|---|
| Load and understand a dataset | `src/data_loader.py` loads and validates Iris (built-in, bundled CSV or uploaded CSV). Dashboard Section 2 and the notebook show size, features, target, types, missing values, duplicates, class distribution, statistics, distributions and correlations. |
| Split data into training and testing sets | `split_data()` in `src/preprocessing.py`: shuffled, stratified, 80/20 by default, fixed seed. Dashboard Section 3. |
| Apply a simple classification algorithm | Logistic Regression (default); Decision Tree and K-Nearest Neighbors (K = 5) also implemented (`src/model_training.py`). |
| Model training | `train_pipeline()` fits imputer, scaler and classifier on training data only; model saved with joblib. Dashboard Section 4. |
| Model evaluation | `src/evaluation.py`: accuracy, precision, recall, F1, confusion matrix, classification report; cross-validation; overfitting and imbalance checks. Dashboard Section 5. |
| Supervised learning | A labelled-data to labelled-prediction workflow: learn from labelled training data, test on unseen labelled data, predict new samples (Section 6). |
| Feature scaling (reference slides) | `StandardScaler` in the pipeline. |
| Shuffle before the split (reference slides) | `shuffle=True` in `split_data()`. |
| Confusion matrix and F1 (reference slides) | Section 5 of the dashboard. |
| KNN with K = 5 and choosing K (reference slides) | Selectable KNN (K = 5) and a cross-validated K-selection chart (Section 4). |
| Accuracy can mislead on imbalanced data (reference slides) | Imbalance warning and per-class metrics; covered by a test. |
| Test with new data (reference slides) | Manual prediction panel (Section 6). |
