# Great Expectations — Medical Appointment Data Validation

This repository is a small, presentation-ready MLOps example built with Great Expectations (GX). It validates a medical appointment dataset before the data is used by downstream analytics or machine-learning workflows.

The project uses one reusable validation pipeline with two versions of the same dataset:

- `medical_appointments_clean.csv`: passes all active Expectations.
- `medical_appointments_bad.csv`: contains deliberate data-quality problems that fail multiple Expectations.

The source CSV files are read but never repaired or overwritten by the demo.

## Validation workflow

```text
CSV -> pandas DataFrame -> GX Batch -> Expectation Suite
    -> Validation Definition -> Checkpoint -> Actions -> Data Docs
```

The Checkpoint runs the Validation Definition and then executes two Actions:

1. `UpdateDataDocsAction` generates or updates the local HTML validation report.
2. `WriteStatusAction` writes a short success or failure message to `artifacts/data_quality_status.txt`.

## Active data-quality rules

The Expectation Suite contains eight active Expectations:

1. The table must contain the expected columns in the expected order.
2. `AppointmentID` values must not be null.
3. `AppointmentID` values must be unique.
4. `Age` must be between 0 and 115 inclusive.
5. `Gender` must contain only `F` or `M`.
6. `Neighbourhood` values must not be null.
7. `No-show` must contain only `No` or `Yes`.
8. `WaitingDays` must be nonnegative.

`WaitingDays` is derived in memory by comparing calendar dates. It is not written back to either source CSV.

An additional `Scholarship` Expectation is inactive in `gx_demo.py`. It can be enabled during the live demo to show how a new rule is added to an existing Suite. The clean dataset passes this rule, while the bad dataset contains `Scholarship = 2` and fails it.

## Setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the demo

Select the dataset near the top of `gx_demo.py` by leaving one filename active:

```python
FILE_NAME = "medical_appointments_clean.csv"
# FILE_NAME = "medical_appointments_bad.csv"
```

Then run:

```bash
python gx_demo.py
```

To switch scenarios, change only `FILE_NAME` and run the same command again.

Expected results with the optional Scholarship rule inactive:

| Dataset | Evaluated | Passed | Failed | Overall |
| --- | ---: | ---: | ---: | --- |
| Clean | 8 | 8 | 0 | PASSED |
| Bad | 8 | 1 | 7 | FAILED |

When the Scholarship rule is enabled, the bad dataset evaluates nine Expectations and fails eight.

## Generated outputs

The first run automatically creates:

```text
artifacts/
└── data_quality_status.txt

gx/
├── expectations/
├── validation_definitions/
└── uncommitted/
    ├── validations/
    └── data_docs/local_site/
```

- `gx/uncommitted/validations/` contains machine-readable Validation Result JSON files.
- `gx/uncommitted/data_docs/local_site/` contains the human-readable HTML report.
- `artifacts/data_quality_status.txt` contains the latest Action outcome.

Open Data Docs on macOS with:

```bash
open gx/uncommitted/data_docs/local_site/index.html
```

Generated GX outputs and the local status file can be reset safely:

```bash
rm -rf gx artifacts
```

They are recreated on the next run.

## Four-minute live-demo sequence

1. Show the selected CSV and the first rows of the data.
2. Run the clean dataset and show the `8/8` passing terminal summary.
3. Open Data Docs and show the successful validation.
4. Change only `FILE_NAME` to the bad dataset and rerun.
5. Show the seven failed Expectations and invalid-value examples.
6. Open Data Docs and compare the clean and bad Validation Results.
7. Enable the Scholarship Expectation and rerun the bad dataset.
8. Show that the evaluated count increases from eight to nine.

## Repository contents

```text
data/
├── medical_appointments_clean.csv
└── medical_appointments_bad.csv
gx_demo.py
presentation.pdf
requirements.txt
README.md
```

The datasets and model context are teaching artifacts. This project demonstrates data validation and does not make a medical prediction or claim clinical validity.
