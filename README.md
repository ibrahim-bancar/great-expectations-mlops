# Great Expectations: Data Validation Before Model Training

This repository is a small, complete MLOps example for a live presentation. It validates a loan-approval dataset with Great Expectations (GX) before a scikit-learn model can train.

The central behavior is a validation gate:

```text
CSV data -> GX expectation suite -> pass -> train model -> save metrics
                               \-> fail -> show violations -> stop
```

The repository contains two deterministic datasets:

- `data/loan_applications_bad.csv` contains six deliberate data-quality problems.
- `data/loan_applications_clean.csv` satisfies the data contract.

## Where GX fits in the ML lifecycle

The demo places data validation between ingestion and training. This catches schema drift, missing values, duplicate identifiers, invalid ranges, and invalid labels before they silently affect features, metrics, or a deployed model.

GX represents the workflow with five core concepts:

1. A **Data Context** manages GX configuration and validation metadata.
2. A **Data Source** connects GX to pandas.
3. A **Data Asset** names the logical dataset.
4. A **Batch Definition** identifies the data validated in one run.
5. An **Expectation Suite** groups the rules that form the data contract.

The code joins the Batch Definition and Expectation Suite in a **Validation Definition**, then runs it with the current dataframe.

## Data contract

The expectation suite checks:

- Exact column names and order
- Unique applicant IDs
- No missing values in required columns
- Age between 18 and 80
- Annual income between 1 and 500,000
- Debt-to-income ratio between 0 and 1
- Credit score between 300 and 850
- Label values limited to 0 or 1

## Setup

GX Core 1.24 supports Python 3.10 through 3.14. Create an isolated environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the live demo

Run the intentionally broken dataset first. The command exits with status 2 after GX reports the failed expectations. The model does not train.

```bash
python -m src.pipeline \
  --data data/loan_applications_bad.csv \
  --output-dir artifacts/bad
```

Then run the clean dataset. All expectations pass and the classifier trains.

```bash
python -m src.pipeline \
  --data data/loan_applications_clean.csv \
  --output-dir artifacts/clean
```

For the full two-part sequence:

```bash
bash scripts/demo.sh
```

Generated files appear under `artifacts/`:

- `validation_result.json` contains the detailed GX validation result.
- `model_metrics.json` appears only after successful validation.

## Recreate the datasets

The committed CSV files are deterministic. To recreate them:

```bash
python scripts/generate_data.py
```

## Run the tests

```bash
pytest -q
```

The tests verify that clean data passes, bad data blocks training, and a successful run writes model metrics.

## Four-minute demo plan

1. Show the two CSV files and point out one bad value.
2. Open `build_expectation_suite()` in `src/pipeline.py` and explain the rules as a data contract.
3. Run the bad dataset. Highlight `VALIDATION FAILED` and `TRAINING BLOCKED`.
4. Run the clean dataset. Highlight `VALIDATION PASSED` and `MODEL TRAINED`.
5. Open the generated JSON files to show that validation evidence is saved.

## Backup for the live demo

If the live environment fails, use the verified output captures in `backup/`:

- `backup/validation-failed.png` shows the six failed expectations and the blocked training step.
- `backup/validation-passed.png` shows all 13 expectations passing and model training completing.

The same captures also appear on slides 7 and 8.

## Project structure

```text
.
├── data/
│   ├── loan_applications_bad.csv
│   └── loan_applications_clean.csv
├── backup/
│   ├── validation-failed.png
│   └── validation-passed.png
├── scripts/
│   ├── demo.sh
│   └── generate_data.py
├── src/
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## References

- [GX Core documentation](https://docs.greatexpectations.io/docs/core/introduction/)
- [Connect to dataframe data](https://docs.greatexpectations.io/docs/core/connect_to_data/dataframes/)
- [Expectation Suites](https://docs.greatexpectations.io/docs/core/define_expectations/organize_expectation_suites/)
- [Validation Definitions](https://docs.greatexpectations.io/docs/core/run_validations/)

