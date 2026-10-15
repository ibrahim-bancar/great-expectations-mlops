#!/usr/bin/env bash
set -u

echo "1) Bad data: validation should fail and training should stop"
python -m src.pipeline --data data/loan_applications_bad.csv --output-dir artifacts/bad
bad_status=$?
if [[ $bad_status -ne 2 ]]; then
  echo "Unexpected exit code for bad data: $bad_status"
  exit 1
fi

echo
echo "2) Clean data: validation should pass and training should run"
python -m src.pipeline --data data/loan_applications_clean.csv --output-dir artifacts/clean

