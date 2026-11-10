# Bike Taxi Demand Prediction

A Python machine-learning project that explores hourly bike-taxi demand, engineers time and weather features, compares regression models, evaluates them on a chronological holdout set, and generates future demand estimates and visual reports.

## What is included

- Reproducible synthetic hourly training data and future inference examples.
- A data-generation script with a fixed random seed.
- Data validation and time-based feature engineering.
- Chronological train, validation and test splits to avoid random time leakage.
- A model comparison workflow using validation MAE.
- A final holdout evaluation using MAE, RMSE, R² and WAPE.
- A saved model bundle, feature-importance output, prediction CSVs and report charts generated when the workflow runs.
- Unit tests for data preparation, splitting, training, persistence and inference.

## Important dataset note

The included CSV files are **synthetic demonstration data**, not real customer trips or operational records. They allow the complete workflow to run after installation, but performance metrics on them do not establish real-world bike-taxi demand accuracy. Use an appropriately licensed real dataset for any real-world performance claims. See [`data/README.md`](data/README.md).

## Project structure

```text
Bike-Taxi-Demand-Prediction/
├── data/
│   ├── sample/                 # Included synthetic training and inference CSVs
│   └── raw/                    # Optional external dataset (not committed)
├── notebooks/                  # Exploratory analysis notebooks
├── src/bike_demand/             # Data, features, training, prediction and reporting modules
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_demo.py              # One-command end-to-end workflow
│   ├── train.py
│   ├── predict.py
│   └── generate_report.py
├── tests/
├── artifacts/                  # Generated model and CSV outputs
└── reports/figures/             # Generated charts
```

## Requirements

- Python 3.10 or newer
- pip

## Install

### Windows PowerShell

Run these commands from the `Bike-Taxi-Demand-Prediction` project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

If the virtual environment already exists, do not recreate it; run only the pip upgrade/install commands as needed.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

## Run the complete demo

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
```

The runner will generate the included synthetic sample files if they are missing, train and compare candidate models, evaluate the selected model on the latest chronological holdout period, predict the next 24 synthetic observations, and generate charts plus a Markdown report.

To regenerate the sample data with the default seed and date range:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_data.py
```

## Run individual steps

```powershell
# Train and compare models
.\.venv\Scripts\python.exe scripts\train.py

# Predict using the included future input rows (requires a trained model bundle)
.\.venv\Scripts\python.exe scripts\predict.py

# Generate charts and report from the training/holdout outputs
.\.venv\Scripts\python.exe scripts\generate_report.py

# Run automated tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Check dependency consistency
.\.venv\Scripts\python.exe -m pip check
```

To train on an external dataset, place an authorized CSV at `data/raw/ola.csv` and pass it explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data data/raw/ola.csv
```

## Generated outputs

- `artifacts/model_bundle.joblib` – fitted preprocessing and model pipeline, excluded from Git by default.
- `artifacts/model_leaderboard.csv` – candidate-model validation comparison.
- `artifacts/test_predictions.csv` – holdout actuals, predictions and residuals.
- `artifacts/future_predictions.csv` – predictions for the future input sample.
- `artifacts/feature_importance.csv` – model feature importance or coefficient magnitudes.
- `reports/model_report.md` – selected model and holdout metrics, with the synthetic-data limitation stated.
- `reports/figures/` – demand, comparison, prediction and residual charts.

## Modeling approach

1. Validate the expected dataset schema and parse timestamps.
2. Create hour, weekday, month, weekend, peak-hour, holiday and cyclical time features.
3. Impute numeric and categorical missing values inside the model pipeline.
4. Split observations chronologically into 70% training, 15% validation and 15% test partitions.
5. Compare baseline and regression models using validation MAE.
6. Refit the selected model on training plus validation data.
7. Evaluate once on the later holdout partition and save predictions and metrics.

The current model does not use lagged or rolling demand features. Those should only be added with a clearly defined prediction horizon and features that would actually be available at prediction time.

## Limitations

- Synthetic sample results are only a demonstration of code execution and evaluation mechanics.
- Real-world usefulness depends on the quality, representativeness, permissions and time coverage of a real dataset.
- The workflow is an analytical project, not a live dispatch or production prediction service.
