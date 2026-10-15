# Great Expectations: Data Reliability in an ML System

This repository is a small, complete MLOps example for a live presentation. It uses Great Expectations (GX) as an executable data-quality gate: invalid loan-application data blocks model training, while a valid batch is allowed to continue.

## Objective

- **Business objective:** prevent invalid input batches from influencing model-development decisions.
- **ML objective:** train a binary classifier on validated loan-application features.
- **System objective:** fail visibly, preserve validation evidence, and make the same contract repeatable in local runs and CI.

The dataset and model are synthetic teaching artifacts; the reported accuracy is not a lending-model quality claim.

## Where GX fits in the ML lifecycle

The lifecycle is iterative rather than a one-way training script:

```text
Business objective -> Data engineering -> GX validation -> Model development
        ^                                                   |
        |                                                   v
Business analysis <- Monitoring <- Deployment <-------------+
```

GX validates a batch at the boundary between data engineering and model development. A failed contract prevents that batch from reaching `model.fit()`.

```text
CSV data -> GX Checkpoint -> pass -> train model -> save metrics
                         \-> fail -> report violations -> stop
```

## How the architecture works

1. A **Data Context** manages GX configuration and validation metadata.
2. A **Data Source** connects GX to pandas.
3. A **Data Asset** names the logical dataset.
4. A **Batch Definition** identifies the data validated in one run.
5. An **Expectation Suite** groups the rules that form the data contract.
6. A **Validation Definition** connects the batch definition to the suite.
7. A **Checkpoint** runs that validation as a reusable workflow boundary and returns structured results.

The demo deliberately uses an ephemeral Data Context so every run is self-contained. A production project normally uses a persistent context and durable validation stores.

## Data contract

The suite contains 13 expectations covering:

- exact column names and order;
- unique applicant IDs;
- no missing values in required columns;
- age between 18 and 80;
- annual income between 1 and 500,000;
- debt-to-income ratio between 0 and 1;
- credit score between 300 and 850;
- label values limited to 0 or 1.

## System qualities from the course

- **Reliability:** invalid batches fail loudly before training.
- **Maintainability:** the contract is readable, tested, and version-controlled with the code.
- **Adaptability:** legitimate domain changes can be reviewed as explicit expectation changes.
- **Scalability:** the same GX concepts can be used with dataframe, file, SQL, and Spark data sources; this demo uses pandas only.

## Validation is not drift monitoring

Validation asks whether the current batch satisfies known rules such as schema, completeness, ranges, and allowed labels. Drift monitoring asks whether distributions, model behavior, or business outcomes changed over time. GX is the pre-training gate in this demo; a production system still needs post-deployment monitoring and feedback into the next iteration.

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

Each run writes:

- `validation_result.json`: detailed GX validation evidence;
- `run_metadata.json`: timestamp, input SHA-256, GX version, row/column counts, suite and Checkpoint names, and pipeline outcome;
- `model_metrics.json`: created only after successful validation and training.

## Recreate the datasets

The committed CSV files are deterministic:

```bash
python scripts/generate_data.py
```

## Run the tests

```bash
pytest -q
```

The tests verify the contract, blocked and successful control flow, and saved metadata. GitHub Actions runs the same test suite on every push and pull request.

## Four-minute demo plan

1. Show the two CSV files and point out one bad value.
2. Open `build_expectation_suite()` and the Checkpoint in `src/pipeline.py`.
3. Run the bad dataset; highlight `VALIDATION FAILED` and `TRAINING BLOCKED`.
4. Run the clean dataset; highlight `VALIDATION PASSED` and `MODEL TRAINED`.
5. Open `run_metadata.json` to show the dataset fingerprint and reproducibility evidence.

## Backup for the live demo

If the live environment fails, use the verified captures in `backup/`:

- `backup/validation-failed.png`
- `backup/validation-passed.png`

## Project structure

```text
.
├── .github/workflows/tests.yml
├── backup/
├── data/
├── scripts/
├── slides/
├── src/pipeline.py
├── tests/test_pipeline.py
├── PRESENTATION_NOTES.md
├── requirements.txt
└── README.md
```

## References

- [GX Core documentation](https://docs.greatexpectations.io/docs/core/introduction/)
- [Connect to data](https://docs.greatexpectations.io/docs/core/connect_to_data/)
- [Expectation Suites](https://docs.greatexpectations.io/docs/core/define_expectations/organize_expectation_suites/)
- [Validation Definitions](https://docs.greatexpectations.io/docs/core/run_validations/)
- [Checkpoints](https://docs.greatexpectations.io/docs/core/run_validations/create_a_checkpoint/)
