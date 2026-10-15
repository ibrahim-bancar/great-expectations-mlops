"""Create deterministic clean and intentionally broken demo datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    row_count = 200
    age = rng.integers(21, 70, size=row_count)
    income = rng.integers(24_000, 180_000, size=row_count)
    dti = np.round(rng.uniform(0.05, 0.75, size=row_count), 3)
    credit = rng.integers(420, 830, size=row_count)

    approval_score = (
        0.009 * (credit - 600)
        + 0.000018 * (income - 60_000)
        - 4.5 * (dti - 0.35)
        - 0.012 * np.abs(age - 40)
        + rng.normal(0, 0.55, size=row_count)
    )
    approved = (approval_score > 0).astype(int)

    clean = pd.DataFrame(
        {
            "applicant_id": [f"A{i:04d}" for i in range(1, row_count + 1)],
            "age": age,
            "annual_income": income,
            "debt_to_income": dti,
            "credit_score": credit,
            "approved": approved,
        }
    )
    bad = clean.copy()
    bad.loc[3, "age"] = 15
    bad.loc[17, "debt_to_income"] = 1.4
    bad.loc[31, "credit_score"] = 950
    bad.loc[48, "annual_income"] = np.nan
    bad.loc[65, "approved"] = 3
    bad.loc[72, "applicant_id"] = bad.loc[71, "applicant_id"]

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    clean.to_csv(data_dir / "loan_applications_clean.csv", index=False)
    bad.to_csv(data_dir / "loan_applications_bad.csv", index=False)
    print(f"Wrote {len(clean)} rows to each demo dataset in {data_dir}/")


if __name__ == "__main__":
    main()

