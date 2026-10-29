from pathlib import Path
from typing import Literal

import great_expectations as gx
import pandas as pd
from great_expectations.checkpoint import Checkpoint
from great_expectations.checkpoint.actions import (
    ActionContext,
    UpdateDataDocsAction,
    ValidationAction,
)
from great_expectations.exceptions import DataContextError


FILE_NAME = "medical_appointments_clean.csv"
# FILE_NAME = "medical_appointments_bad.csv"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / FILE_NAME
STATUS_FILE = PROJECT_DIR / "artifacts" / "data_quality_status.txt"
CHECKPOINT_NAME = "medical_appointments_checkpoint"




# Step 1: The Data Context manages GX configuration, metadata, and Data Docs.
# Create the central GX project object.
context = gx.get_context(mode="file", project_root_dir=PROJECT_DIR)
context.variables.progress_bars = {
    "globally": False,
    "metric_calculations": False,
}





# Load the selected CSV and derive a field used for validation.
dataframe = pd.read_csv(DATA_PATH)


# Compute a derived column for the number of days between scheduling and appointment.
scheduled_dates = pd.to_datetime(
    dataframe["ScheduledDay"], utc=True, errors="coerce"
).dt.normalize()

appointment_dates = pd.to_datetime(
    dataframe["AppointmentDay"], utc=True, errors="coerce"
).dt.normalize()

dataframe["WaitingDays"] = (appointment_dates - scheduled_dates).dt.days


# Step 2: Connect the selected CSV to a Data Source, Asset, and Batch Definition.
# The Data Source connects GX to pandas.
data_source = context.data_sources.add_or_update_pandas(
    name="medical_appointments_source"
)

# The Data Asset represents the logical appointments dataset.
data_asset = data_source.add_dataframe_asset(name="medical_appointments_asset")

# The Batch Definition selects the whole DataFrame for validation.
batch_definition = data_asset.add_batch_definition_whole_dataframe(
    name="selected_csv_batch"
)

# Batch Parameters provide the current in-memory DataFrame at runtime.
batch_parameters = {"dataframe": dataframe}

# The Batch is the concrete data validated in this run.
batch = batch_definition.get_batch(batch_parameters=batch_parameters)




# Step 3: Each Expectation is one rule; the Suite is the complete data contract.


# The Suite groups all Expectations into one data contract.
suite = gx.ExpectationSuite(name="medical_appointments_suite")

expected_columns = [
    "PatientId",
    "AppointmentID",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show",
    "WaitingDays",
]

suite.add_expectation(
    gx.expectations.ExpectTableColumnsToMatchOrderedList(
        column_list=expected_columns,
        description="The batch must contain the expected columns in order.",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="AppointmentID",
        description="Every appointment must have an ID.",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(
        column="AppointmentID",
        description="Each appointment ID must be unique.",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="Age",
        min_value=0,
        max_value=115,
        description="Age must be between 0 and 115 inclusive.",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="Gender",
        value_set=["F", "M"],
        description="Gender must use the values present in the source contract.",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="Neighbourhood",
        description="Neighbourhood is required for every appointment.",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="No-show",
        value_set=["No", "Yes"],
        description="The target label must be No or Yes.",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="WaitingDays",
        min_value=0,
        description="An appointment cannot occur before it is scheduled.",
    )
)

# Total 8 Expectations are defined.


# suite.add_expectation(
#     gx.expectations.ExpectColumnValuesToBeInSet(
#         column="Scholarship",
#         value_set=[0, 1],
#         description="Scholarship must be represented as 0 or 1.",
#     )
# )


# Replacing the stored Suite prevents a previously uncommented rule from lingering.
# Save the current Suite in the Data Context.
suite = context.suites.add_or_update(suite)


# Step 4: The Validation Definition connects the Batch Definition and Suite.
# Define which Batch Definition is checked by which Suite.
validation_definition = gx.ValidationDefinition(
    name="validate_medical_appointments",
    data=batch_definition,
    suite=suite,
)
validation_definition = context.validation_definitions.add_or_update(
    validation_definition
)


# Define a custom Action that records the Checkpoint outcome.
class WriteStatusAction(ValidationAction):
    """Write a local status file after the Checkpoint produces its results."""

    type: Literal["write_status_file"] = "write_status_file"
    status_file: str

    def run(
        self,
        checkpoint_result,
        action_context: ActionContext | None = None,
    ) -> dict:
        message = (
            "Data quality check passed"
            if checkpoint_result.success
            else "Data quality issues detected"
        )
        output_path = Path(self.status_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(message + "\n", encoding="utf-8")
        return {"message": message, "status_file": str(output_path)}






# Remove a temporary record left by an interrupted or previous demo run.
try:
    context.checkpoints.delete(CHECKPOINT_NAME)
except DataContextError:
    pass

# Step 4 continued: The Checkpoint runs validation, actions, and Data Docs.

# The Checkpoint combines validations with post-validation Actions.
checkpoint = Checkpoint(
    name=CHECKPOINT_NAME,
    validation_definitions=[validation_definition],
    actions=[
        # Built-in Action: update the Data Docs report.
        UpdateDataDocsAction(
            name="update_local_data_docs",
            site_names=["local_site"],
        ),
        # Custom Action: write the validation status to a local file.
        WriteStatusAction(
            name="write_local_status",
            status_file=str(STATUS_FILE),
        ),
    ],
    result_format={
        "result_format": "SUMMARY",
        "partial_unexpected_count": 10,
    },
)

# Run the workflow with the current DataFrame.
checkpoint_result = checkpoint.run(batch_parameters=batch_parameters)
context.checkpoints.delete(CHECKPOINT_NAME)  # Prevent conflicts on repeated runs.


# Collect the Validation Results produced by the Checkpoint.
validation_results = list(checkpoint_result.run_results.values())

# This demo expects exactly one Validation Result.
if len(validation_results) != 1:
    raise RuntimeError("Expected exactly one Validation Result.")

# Select the single Validation Result for reporting.
validation_result = validation_results[0]

# Read the summary statistics and keep only failed Expectations.
statistics = validation_result.statistics
failed_results = [result for result in validation_result.results if not result.success]

# Read the message written by the custom Action.
action_message = STATUS_FILE.read_text(encoding="utf-8").strip()

# Get the local Data Docs URL for the terminal summary.
docs_sites = context.get_docs_sites_urls(site_names=["local_site"])
docs_url = docs_sites[0]["site_url"] if docs_sites else "Not available"



# Display Summary
print("\nGreat Expectations validation summary")
print(f"Dataset: {FILE_NAME}")
print(f"Batch retrieved: {batch.id}")
print(f"Rows: {len(dataframe):,}")
print(f"Expectations evaluated: {statistics['evaluated_expectations']}")
print(f"Expectations passed: {statistics['successful_expectations']}")
print(f"Expectations failed: {statistics['unsuccessful_expectations']}")
print(f"Overall validation: {'PASSED' if validation_result.success else 'FAILED'}")

if failed_results:
    print("Failed Expectations:")
    for result in failed_results:
        config = result.expectation_config
        column = config.kwargs.get("column", "<table>")
        count = result.result.get("unexpected_count", "n/a")
        examples = result.result.get("partial_unexpected_list", [])
        print(f"  - {config.type} [{column}]: {count} unexpected; examples={examples}")
else:
    print("Failed Expectations: none")

print("Checkpoint execution: COMPLETED")
print(f"Action outcome: {action_message}")
print(f"Data Docs: {docs_url}")

# Uncomment this line to open the updated Data Docs in the default browser.
# context.open_data_docs(site_name="local_site")
