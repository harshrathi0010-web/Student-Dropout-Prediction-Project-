# Student Dropout Prediction

A student-risk prediction project that explores academic and personal factors associated with dropout and trains a binary classification model. It includes a Jupyter notebook for exploration, a scikit-learn training pipeline, and a Streamlit interface prototype.

> **Project status:** This is an educational prototype, not a validated student-support or decision-making system. The Streamlit prediction flow currently has input-mapping issues and needs to be corrected and tested before its predictions can be relied on.

## Problem

Students who may be at risk of dropping out can be difficult to identify early. This project uses available student data, such as attendance, study habits, grades, assignment delays, and other recorded characteristics, to estimate the `Dropout` target.

The intention is to support further analysis and exploration. A model output should not be treated as a definitive assessment of a student or used on its own to make decisions about them.

## How it works

The training script in `train.py`:

1. Reads `data/student.csv` and separates the predictors from the `Dropout` target.
2. Removes the `Student_ID` identifier and caps `Family_Income` values at 100,000.
3. Builds preprocessing pipelines:
   - Categorical values: fill missing values with the most frequent value, then one-hot encode.
   - Numeric values: fill missing values with the mean, then standardize.
4. Combines preprocessing with a logistic regression classifier.
5. Splits the data into training and test sets (80/20), trains the model, and evaluates it.
6. Saves the trained pipeline to `models/dropout_models.pkl` and evaluation metrics to `models/model_evaluation.txt`.

The Streamlit interface in `app.py` collects student characteristics and is intended to pass them to the saved model for a prediction.

## Project flow

```text
data/student.csv
       |
       +--> notebook/EDA1.ipynb (explore the data)
       |
       +--> train.py (preprocess, train, evaluate)
                    |
                    +--> models/dropout_models.pkl
                    +--> models/model_evaluation.txt
                               |
                               +--> app.py (Streamlit UI prototype)
```

## Results

The checked-in evaluation report records **81% accuracy** on a 2,000-row test split. For class `1`, the report shows **0.42 recall** and **0.52 F1-score**. In other words, the model missed many of the class-1 examples in this test split; accuracy alone gives an incomplete picture.

Confirm the dataset's exact meaning for target values `0` and `1` before interpreting or presenting predictions.

## Run locally

From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train and evaluate the model:

```powershell
python train.py
```

Start the Streamlit interface:

```powershell
streamlit run app.py
```

Training expects `data/student.csv` and a writable `models/` directory. The Streamlit interface expects the saved model artifact and dataset at those project-relative paths.

## Limitations

- The dataset has 10,000 rows and contains missing values; it may not represent other institutions, student populations, or time periods.
- Evaluation is based on a single random train/test split. There is no cross-validation or independent external validation.
- The reported recall for class `1` is 0.42. If class `1` represents dropout, this means many dropout cases were not identified in the test set.
- Only logistic regression is evaluated. The saved metrics do not establish that it is the best or a sufficiently reliable model.
- The interface currently does not reliably map all collected values to the model's expected feature names and order. The assignment-delay input is referenced under a different variable name, and the travel-time value is formed as a tuple. The UI also displays the raw predicted class rather than a calibrated risk estimate.
- No fairness, privacy, calibration, or real-world impact assessment is included.
- A saved model can become incompatible with changed data columns or library versions; the model and its training environment are not versioned together.

## Recommended improvements

1. **Fix and test the app-to-model contract.** Build a single-row DataFrame with the exact training feature names and order; correct the assignment-delay and travel-time inputs; load the artifact with `joblib.load`; add an end-to-end prediction test.
2. **Improve evaluation.** Use a stratified split, cross-validation, and metrics such as class-specific recall, precision, F1, ROC-AUC, and a confusion matrix. Choose thresholds based on the intended support use, not accuracy alone.
3. **Check data quality and provenance.** Document the dataset source, label definitions, collection period, missingness, and any restrictions on use.
4. **Assess fairness and privacy.** Check performance across appropriate student groups, minimize personal data, and restrict access to predictions.
5. **Make results actionable and responsible.** Show uncertainty and clear explanations where feasible, and ensure predictions lead only to supportive human review, not automatic penalties or decisions.
6. **Strengthen reproducibility.** Record dependency versions, data/model versions, and evaluation settings; add automated tests for preprocessing and prediction.

## Repository structure

```text
.
|-- app.py                       # Streamlit interface prototype
|-- train.py                     # Preprocessing, training, and evaluation
|-- requirements.txt             # Python dependencies
|-- data/
|   |-- student.csv              # Training dataset
|   `-- student_cleaned.csv      # Cleaned dataset
|-- models/
|   |-- dropout_models.pkl       # Saved scikit-learn pipeline
|   `-- model_evaluation.txt     # Test-split metrics
`-- notebook/
    `-- EDA1.ipynb               # Exploratory data analysis
```
