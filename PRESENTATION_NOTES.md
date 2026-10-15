# Presentation notes

Target timing: 6 minutes for the slides, 4 minutes for the live demo, then Q&A.

## Slide 1 — Great Expectations: Data Reliability in ML Systems

“My topic is Great Expectations. I will present it not only as a data-checking library, but as a reliability component inside an ML system. The demo turns assumptions about training data into an executable gate.”

## Slide 2 — Silent data failures in production

“A pipeline can complete successfully while the data is wrong. Missing values, impossible ranges, duplicates, or schema changes may not raise a Python error, but they can change features and model behavior. This is a systems problem, not only a modeling problem.”

## Slide 3 — Business, ML, and system objectives

“The business objective is to prevent invalid batches from influencing development decisions. The ML objective is a small binary classifier. The system objective is to fail visibly and preserve evidence. Starting from these objectives keeps the demo focused on operational reliability rather than the accuracy number.”

## Slide 4 — GX in the iterative ML lifecycle

“The lifecycle is iterative: business analysis informs data work; validated data reaches model development; deployment creates monitoring signals; those signals feed the next iteration. GX sits between data engineering and model development. It validates known rules before training, but it does not replace post-deployment drift or model monitoring.”

## Slide 5 — Core concepts and Checkpoint architecture

“The Data Context manages GX objects. The Data Source and Data Asset identify the logical data. A Batch Definition selects the current batch. An Expectation Suite contains the contract. A Validation Definition binds data to the suite, and a Checkpoint runs that validation as a reusable workflow boundary.”

## Slide 6 — The executable data contract

“The suite contains thirteen checks for schema, identifiers, completeness, ranges, and labels. The Checkpoint produces a structured result. Our code branches on that result: a failed batch returns a nonzero exit code before model training.”

## Slide 7 — Failed batch: training blocked

“The bad file contains six deliberate violations. GX reports the affected rule, column, count, and examples. The pipeline saves the detailed result and run metadata, including an input hash, then stops. No model metric is produced.”

## Slide 8 — Valid batch: training continues

“The clean file passes all thirteen checks. Only then does the classifier train. The accuracy only demonstrates that the next pipeline stage ran. The useful evidence is the validated result plus the dataset fingerprint and tool version.”

## Slide 9 — MLOps system qualities

“This gate improves reliability by failing loudly, maintainability by keeping rules readable and version-controlled, and adaptability by making legitimate contract changes explicit. The same concept can scale to other GX backends, although this small demo intentionally uses pandas.”

## Slide 10 — Boundary and takeaway

“Validation checks whether a batch meets known expectations. Monitoring detects changes over time in data, model behavior, and business outcomes. Great Expectations solves the first problem here. The takeaway is that data assumptions become executable controls in the ML lifecycle.”

## Four-minute live demo

1. Inspect one invalid row and `build_expectation_suite()` — 45 seconds.
2. Show the Checkpoint and the branch on `result.success` — 45 seconds.
3. Run bad data and show the blocked path plus `run_metadata.json` — 70 seconds.
4. Run clean data and show the pass path plus metrics — 70 seconds.
5. Summarize the validation/monitoring boundary — 30 seconds.

## Likely Q&A

**How is this different from unit tests?**

Unit tests usually check code behavior with controlled inputs. GX evaluates properties of an actual data batch and produces structured validation evidence.

**Does GX clean the data?**

No. It detects and reports violations. The pipeline decides whether to block, quarantine, alert, or trigger remediation.

**Does this detect drift?**

Not by itself in this demo. These expectations enforce known constraints. Distribution and model drift require reference windows, time-based comparison, monitoring, and alerting.

**Why use an ephemeral Data Context?**

It keeps the classroom demo self-contained. A production project normally uses persistent configuration and validation stores.

## Course alignment

- W01: production ML systems include a data stack, assumptions must be monitored, production data is messy, and data needs testing/versioning.
- W02: begin with business and ML objectives; design for reliability, scalability, maintainability, and adaptability; manage artifacts for reproducibility; treat the lifecycle as iterative; monitor deployment outcomes.
