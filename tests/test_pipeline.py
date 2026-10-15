import hashlib
import json
from pathlib import Path

import pandas as pd

from src.pipeline import run_pipeline, validate_dataframe


def test_clean_data_passes_validation() -> None:
    dataframe = pd.read_csv("data/loan_applications_clean.csv")
    result = validate_dataframe(dataframe)
    assert result.success is True
    assert result.statistics["evaluated_expectations"] == 13


def test_bad_data_fails_and_blocks_training(tmp_path: Path) -> None:
    output_dir = tmp_path / "bad-run"
    exit_code = run_pipeline(
        Path("data/loan_applications_bad.csv"), output_dir
    )
    assert exit_code == 2
    assert (output_dir / "validation_result.json").exists()
    metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["validation_success"] is False
    assert metadata["pipeline_outcome"] == "training_blocked"
    assert not (output_dir / "model_metrics.json").exists()


def test_clean_data_trains_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "clean-run"
    exit_code = run_pipeline(
        Path("data/loan_applications_clean.csv"), output_dir
    )
    assert exit_code == 0
    assert (output_dir / "validation_result.json").exists()
    assert (output_dir / "model_metrics.json").exists()
    metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    expected_hash = hashlib.sha256(
        Path("data/loan_applications_clean.csv").read_bytes()
    ).hexdigest()
    assert metadata["input_sha256"] == expected_hash
    assert metadata["great_expectations_version"] == "1.24.0"
    assert metadata["checkpoint_name"] == "training_data_quality_gate"
    assert metadata["pipeline_outcome"] == "training_completed"

