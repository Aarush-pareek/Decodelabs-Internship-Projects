# AI Data Classification System

An end-to-end **supervised-learning** project with a Streamlit analytics dashboard. It loads a dataset, explores and cleans it, splits it into training and testing sets, trains a classifier, evaluates it with proper metrics, and predicts new samples while explaining *why* the model chose a class.

Built for **Internship Project 2: Data Classification Using AI**.

## Project Overview

The application walks through the full classification pipeline in one dashboard:

```
Dataset -> Exploration -> Cleaning -> Feature/Target selection -> Train/Test split
        -> Training -> Evaluation -> Visualisation -> Prediction on new data
```

The default dataset is the classic **Iris** benchmark (150 samples, 4 features, 3 balanced classes). The code is written for any numeric-feature CSV, so you can upload your own file from the sidebar.

## Internship Objective

> Build a basic classification model using a small dataset.
> Key requirements: load and understand a dataset, split data into training and testing sets, apply a simple classification algorithm.
> Key skills: data handling, supervised-learning basics, model training.

See [Internship Requirement Mapping](#internship-requirement-mapping) for exactly where each requirement is implemented.

## Features

- **Dataset explorer**: preview, statistics, data types, missing values, duplicates, class balance, feature distributions, correlation heatmap.
- **Validation and cleaning**: friendly errors for missing files, invalid CSVs, missing columns, too few rows, single-class targets; duplicate and unlabelled rows removed; missing feature values imputed.
- **Configurable split**: choose the features and the test size (10-50%); stratified, shuffled and reproducible.
- **Three algorithms**: Logistic Regression (default), Decision Tree, K-Nearest Neighbors, with an optional side-by-side comparison.
- **Evaluation dashboard**: accuracy, precision, recall, F1, confusion matrix, per-class classification report, plain-language metric explanations, overfitting check, imbalance warning.
- **Choosing K**: cross-validated K-selection chart for K-Nearest Neighbors.
- **Prediction panel**: enter feature values, get the predicted class, the confidence, class probabilities and an explanation of the decision. Out-of-range inputs trigger a warning.
- **Model persistence**: preprocessing and model are saved together in one file and can be reloaded.
- **Automated tests**: 84 pytest tests covering data, preprocessing, models, prediction, charts and UI components.

## Technology Stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, NumPy |
| Machine learning | scikit-learn (Pipeline, StandardScaler, SimpleImputer, LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier) |
| Visualisation | matplotlib, seaborn |
| Persistence | joblib |
| Interface | Streamlit |
| Testing | pytest |

## Machine Learning Workflow

| Stage | What happens | Code |
|---|---|---|
| 1. Load | Built-in Iris, the bundled CSV, or an uploaded CSV; structural validation | `src/data_loader.py` |
| 2. Explore | Shape, types, missing values, duplicates, class balance, statistics | `src/data_loader.py`, `utils/visualization.py` |
| 3. Clean | Drop unlabelled and duplicate rows | `src/preprocessing.py` |
| 4. Select | Choose numeric feature columns and the target column | `src/preprocessing.py` |
| 5. Split | Shuffled, stratified train/test split with a fixed random seed | `src/preprocessing.py` |
| 6. Train | `Pipeline`: median imputation -> standardisation -> classifier | `src/model_training.py` |
| 7. Evaluate | Metrics on the held-out test set, cross-validation on the training set | `src/evaluation.py` |
| 8. Predict | Validate input, predict, report confidence and reasons | `src/prediction.py` |

### Avoiding data leakage

Anything that *learns from data* (the median used to fill gaps, the mean and standard deviation used for scaling) lives **inside the scikit-learn Pipeline** and is fitted on the training rows only. The test set is never used to fit or to choose anything. When models are compared, the comparison uses cross-validation on the training set. A test (`test_scaler_learns_from_training_data_only`) checks this directly.

### Why standardise?

Logistic Regression and K-Nearest Neighbors are both sensitive to feature scale (KNN works on distances). Standardisation gives every feature mean 0 and standard deviation 1 so no feature dominates only because of its units.

## Dataset

**Iris** (Fisher, 1936): 150 flower measurements in centimetres.

| Property | Value |
|---|---|
| Features | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` |
| Target | `species` (setosa, versicolor, virginica) |
| Samples | 150 (50 per class), 149 after removing one exact duplicate |
| Missing values | none |

The same data is available as `data/iris.csv` to demonstrate CSV loading. To use your own file, choose **Upload your own CSV** in the sidebar: numeric feature columns plus one label column (2 to 8 classes, at least 20 labelled rows).

## Algorithms Used

| Algorithm | Idea | Role |
|---|---|---|
| **Logistic Regression** | Learns one weight per feature and turns the weighted sum into class probabilities | **Default / primary** |
| Decision Tree (max depth 4) | Learns a sequence of if/else questions | Comparison |
| K-Nearest Neighbors (K = 5) | Majority vote of the 5 closest training samples | Comparison |

Logistic Regression is the default because it is fast, stable, produces probabilities and its weights can be inspected, which powers the "Why this prediction?" panel. K-Nearest Neighbors with K = 5 is fully implemented and selectable; to make it the default, change `DEFAULT_MODEL` in `src/config.py`.

## Model Evaluation

| Metric | Meaning in plain language |
|---|---|
| Accuracy | Percentage of all predictions that were correct |
| Precision | Of the samples predicted as a class, how many really were that class (few false alarms) |
| Recall | Of the samples that really belong to a class, how many were found (few misses) |
| F1 score | The balance (harmonic mean) of precision and recall |
| Confusion matrix | Rows = true class, columns = predicted class; the diagonal is correct |

Precision, recall and F1 are macro-averaged (every class counts equally). Accuracy alone can mislead on imbalanced data, so the dashboard warns when classes are imbalanced and always shows the per-class report.

## Project Structure

```
ai-data-classification-system/
├── app.py                    Streamlit dashboard (UI only; no ML logic)
├── data/
│   └── iris.csv              Sample dataset demonstrating CSV loading
├── models/
│   └── .gitkeep              trained_model.pkl is generated here after training (git-ignored)
├── src/
│   ├── config.py             Paths, random seed, split limits, defaults
│   ├── data_loader.py        Loading, validation and dataset profiling
│   ├── preprocessing.py      Cleaning, feature/target selection, train/test split
│   ├── model_training.py     Pipelines, training, cross-validation, comparison, save/load
│   ├── evaluation.py         Metrics, confusion matrix, classification report
│   ├── prediction.py         Input validation, prediction, explanations
│   └── workflow.py           Runs the whole pipeline in one call
├── utils/
│   ├── visualization.py      All charts (consistent, accessible colour palette)
│   └── html_components.py    Dashboard cards and styles (HTML-escaped)
├── notebooks/
│   └── exploration.ipynb     Step-by-step walkthrough using the same modules
├── tests/                    pytest suite (84 tests)
├── docs/screenshots/         Place dashboard screenshots here
├── .streamlit/config.toml    Theme and upload-size limit
├── requirements.txt          Runtime dependencies
├── requirements-dev.txt      Runtime + pytest
├── pytest.ini                Test configuration
├── PROJECT_REPORT.md         Detailed project report
├── LICENSE
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd ai-data-classification-system

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## How to Run

```bash
streamlit run app.py
```

Then, in the browser:

1. Pick a dataset in the sidebar (Iris is selected by default).
2. Explore the data in **Section 2**.
3. Choose the features and test-set size in **Section 3**.
4. Press **Train model** in the sidebar.
5. Review training results (**Section 4**) and evaluation (**Section 5**).
6. Enter feature values in **Section 6** and press **Predict**.

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Open `notebooks/exploration.ipynb` in VS Code or Jupyter for the step-by-step walkthrough (select the virtual environment as the kernel).

## Screenshots

> Add screenshots after running the app. Save them in `docs/screenshots/` and uncomment the lines below.

<!--
![Project overview](docs/screenshots/01-overview.png)
![Dataset explorer](docs/screenshots/02-explorer.png)
![Preprocessing](docs/screenshots/03-preprocessing.png)
![Training and comparison](docs/screenshots/04-training.png)
![Evaluation](docs/screenshots/05-evaluation.png)
![Prediction](docs/screenshots/06-prediction.png)
-->

| Placeholder | Suggested capture |
|---|---|
| `01-overview.png` | Section 1 with the trained status |
| `02-explorer.png` | Section 2, class balance and correlation tabs |
| `03-preprocessing.png` | Section 3 with the split summary |
| `04-training.png` | Section 4 with the algorithm comparison |
| `05-evaluation.png` | Section 5 with metrics and confusion matrix |
| `06-prediction.png` | Section 6 with a prediction and its explanation |

## Results

Default configuration: Logistic Regression, Iris, 80/20 stratified split, random seed 42 (119 training rows, 30 test rows).

| Metric | Value |
|---|---|
| Training accuracy | 95.8% |
| **Test accuracy** | **93.3%** (28 of 30 correct) |
| Precision / Recall / F1 (macro) | 93.3% / 93.3% / 93.3% |
| 5-fold cross-validation (training set) | 96.6% (+/- 5.1%) |

Per-class results on the test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| setosa | 1.000 | 1.000 | 1.000 | 10 |
| versicolor | 0.900 | 0.900 | 0.900 | 10 |
| virginica | 0.900 | 0.900 | 0.900 | 10 |

Setosa is perfectly separable. The only two errors are one versicolor predicted as virginica and one virginica predicted as versicolor, the two classes whose measurements overlap.

Algorithm comparison on the same split:

| Model | CV accuracy (train) | Test accuracy |
|---|---|---|
| Logistic Regression | 96.6% | 93.3% |
| Decision Tree | 93.2% | 93.3% |
| K-Nearest Neighbors | 95.7% | 93.3% |

**How much to trust one number.** The test set has only 30 samples, so a single mistake moves accuracy by 3.3 percentage points. Repeating the experiment over 50 different random splits gives a more honest picture:

| Model | Mean test accuracy | Std. dev. | Min | Max |
|---|---|---|---|---|
| Logistic Regression | 95.3% | 3.4% | 86.7% | 100% |
| Decision Tree | 94.6% | 3.1% | 86.7% | 100% |
| K-Nearest Neighbors | 94.7% | 3.3% | 86.7% | 100% |

The three models are statistically indistinguishable on Iris. The seed-42 result (93.3%) is slightly below the 50-split average and was **not** cherry-picked.

## Future Improvements

- Hyper-parameter tuning with `GridSearchCV` (regularisation strength, tree depth).
- Repeated or stratified k-fold evaluation shown in the dashboard instead of a single hold-out split.
- Categorical feature support (one-hot encoding in the pipeline).
- ROC curves and probability calibration for binary problems.
- Batch prediction from an uploaded CSV with downloadable results.
- Experiment history (compare several training runs).
- Containerisation and cloud deployment.

## Internship Requirement Mapping

| Internship Requirement | Implementation |
|---|---|
| Load and understand a dataset | `src/data_loader.py` loads Iris (built-in, bundled CSV, or uploaded CSV) and validates it. Section 2 shows rows, columns, feature names, target, data types, missing values, duplicates, class distribution, statistics, feature distributions and a correlation heatmap. |
| Split data into training and testing sets | `split_data()` in `src/preprocessing.py`: shuffled, stratified `train_test_split`, default 80/20 (adjustable 10-50%), fixed `random_state`. Section 3 shows the resulting sizes. |
| Apply a simple classification algorithm | Logistic Regression (default). Decision Tree and K-Nearest Neighbors (K = 5) are also implemented in `src/model_training.py`. |
| Model training | `train_pipeline()` fits `Imputer -> StandardScaler -> classifier` on the training set only. Section 4 shows status, training accuracy and testing accuracy. The pipeline is saved with joblib. |
| Model evaluation | `src/evaluation.py`: accuracy, precision, recall, F1, confusion matrix and per-class classification report on the held-out test set. Section 5. |
| Supervised learning | Labelled data (features -> known class) is used to learn a mapping, then tested on unseen labelled data and applied to new samples (Section 6). |
| Feature scaling (reference slides) | `StandardScaler` (mean 0, variance 1) inside the pipeline. |
| Shuffle before splitting (reference slides) | `shuffle=True` in `split_data()`. |
| Confusion matrix and F1 (reference slides) | Section 5, with plain-language explanations. |
| K-Nearest Neighbors, K = 5 and choosing K (reference slides) | KNN with `n_neighbors=5` is selectable; Section 4 shows a cross-validated K-selection chart. |
| Accuracy can mislead on imbalanced data (reference slides) | Imbalance warning plus per-class precision, recall and F1. |
| Test with completely new data (reference slides) | Section 6 predicts manually entered samples. |

The reference slides use K-Nearest Neighbors as their example algorithm. This project defaults to Logistic Regression and implements KNN (K = 5) as a selectable and comparable model.

## Author

- **Name:** [Your Name]
- **GitHub:** [github.com/your-username](https://github.com/your-username)
- **LinkedIn:** [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- **Email:** your.email@example.com

## License

Released under the MIT License. See [LICENSE](LICENSE).
