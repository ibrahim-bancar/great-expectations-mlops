"""Validate loan data with Great Expectations before model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = [
    "applicant_id",
    "age",
    "annual_income",
    "debt_to_income",
    "credit_score",
    "approved",
]
FEATURE_COLUMNS = ["age", "annual_income", "debt_to_income", "credit_score"]


def build_expectation_suite() -> gx.ExpectationSuite:
    """Return the data contract used at the training boundary."""
    suite = gx.ExpectationSuite(name="loan_applications_contract")
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=REQUIRED_COLUMNS
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="applicant_id")
    )

    for column in REQUIRED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age", min_value=18, max_value=80
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="annual_income", min_value=1, max_value=500_000
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="debt_to_income", min_value=0, max_value=1
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="credit_score", min_value=300, max_value=850
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="approved", value_set=[0, 1]
        )
    )
    return suite


def validate_dataframe(dataframe: pd.DataFrame):
    """Run the full expectation suite against one in-memory batch."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {
        "globally": False,
        "metric_calculations": False,
    }
    data_source = context.data_sources.add_pandas(name="loan_source")
    data_asset = data_source.add_dataframe_asset(name="loan_applications")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "current_training_batch"
    )

    suite = context.suites.add(build_expectation_suite())
    validation_definition = gx.ValidationDefinition(
        name="validate_before_training",
        data=batch_definition,
        suite=suite,
    )
    validation_definition = context.validation_definitions.add(
        validation_definition
    )
    return validation_definition.run(
        batch_parameters={"dataframe": dataframe},
        result_format={"result_format": "SUMMARY"},
    )


def failed_expectations(validation_result) -> list[dict[str, object]]:
    """Convert failed GX results into concise, serializable diagnostics."""
    failures: list[dict[str, object]] = []
    for item in validation_result.results:
        if item.success:
            continue
        config = item.expectation_config
        result = item.result
        failures.append(
            {
                "expectation": config.type,
                "column": config.kwargs.get("column", "<table>"),
                "unexpected_count": result.get("unexpected_count"),
                "unexpected_percent": result.get("unexpected_percent"),
                "examples": result.get("partial_unexpected_list", []),
            }
        )
    return failures


def train_model(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Train a small classifier after the validation gate passes."""
    features = dataframe[FEATURE_COLUMNS]
    target = dataframe["approved"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    model = LogisticRegression(max_iter=1_000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "rows_used": int(len(dataframe)),
        "test_rows": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
    }


def run_pipeline(csv_path: Path, output_dir: Path) -> int:
    """Validate a CSV, stop on failure, otherwise train and save metrics."""
    dataframe = pd.read_csv(csv_path)
    print(f"Loaded {len(dataframe)} rows from {csv_path}")
    print("Running Great Expectations validation...")
    validation_result = validate_dataframe(dataframe)

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_payload = validation_result.to_json_dict()
    (output_dir / "validation_result.json").write_text(
        json.dumps(validation_payload, indent=2), encoding="utf-8"
    )

    if not validation_result.success:
        failures = failed_expectations(validation_result)
        print(f"VALIDATION FAILED: {len(failures)} expectation(s) failed")
        for failure in failures:
            print(
                "  - {expectation} [{column}]: {count} unexpected; examples={examples}".format(
                    expectation=failure["expectation"],
                    column=failure["column"],
                    count=failure["unexpected_count"],
                    examples=failure["examples"],
                )
            )
        print("TRAINING BLOCKED: fix the data and run the pipeline again")
        return 2

    print(
        "VALIDATION PASSED: "
        f"{validation_result.statistics['successful_expectations']}/"
        f"{validation_result.statistics['evaluated_expectations']} expectations succeeded"
    )
    metrics = train_model(dataframe)
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"MODEL TRAINED: accuracy={metrics['accuracy']:.3f} "
        f"on {metrics['test_rows']} held-out rows"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate loan data before training a classifier."
    )
    parser.add_argument("--data", type=Path, required=True, help="Input CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for validation results and model metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_pipeline(args.data, args.output_dir))


if __name__ == "__main__":
    main()

