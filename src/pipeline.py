"""Validate loan data with Great Expectations before model training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
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
    """Run a GX Checkpoint against one in-memory batch."""
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
    checkpoint = context.checkpoints.add(
        gx.Checkpoint(
            name="training_data_quality_gate",
            validation_definitions=[validation_definition],
            result_format={"result_format": "SUMMARY"},
        )
    )
    checkpoint_result = checkpoint.run(
        batch_parameters={"dataframe": dataframe},
    )
    validation_results = list(checkpoint_result.run_results.values())
    if len(validation_results) != 1:
        raise RuntimeError(
            "Expected exactly one validation result from the Checkpoint"
        )
    return validation_results[0]


def sha256_file(path: Path) -> str:
    """Return a reproducible fingerprint for an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(
    *,
    csv_path: Path,
    dataframe: pd.DataFrame,
    validation_result,
    output_dir: Path,
    pipeline_outcome: str,
) -> None:
    """Persist compact lineage and outcome metadata for this pipeline run."""
    statistics = validation_result.statistics
    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(csv_path),
        "input_sha256": sha256_file(csv_path),
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "great_expectations_version": version("great-expectations"),
        "checkpoint_name": "training_data_quality_gate",
        "expectation_suite_name": "loan_applications_contract",
        "evaluated_expectations": int(statistics["evaluated_expectations"]),
        "successful_expectations": int(statistics["successful_expectations"]),
        "validation_success": bool(validation_result.success),
        "pipeline_outcome": pipeline_outcome,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
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
        write_run_metadata(
            csv_path=csv_path,
            dataframe=dataframe,
            validation_result=validation_result,
            output_dir=output_dir,
            pipeline_outcome="training_blocked",
        )
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
    write_run_metadata(
        csv_path=csv_path,
        dataframe=dataframe,
        validation_result=validation_result,
        output_dir=output_dir,
        pipeline_outcome="training_completed",
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

