# Presentation notes

Target timing: 6 minutes for the slides, 4 minutes for the live demo, then Q&A.

## Slide 1 — Great Expectations: Data Validation

“My topic is Great Expectations, an open-source tool for expressing and running data-quality checks. I will show where it fits in an ML pipeline, how its core objects work together, and a small example that prevents a model from training on invalid data.”

## Slide 2 — The problem: bad data can produce a valid-looking model

“Model code can run successfully even when input data is wrong. A new null, a value outside the valid range, a duplicate record, or a changed label can alter training without producing a Python exception. The pipeline needs an explicit data contract before training starts.”

## Slide 3 — Validation in the ML lifecycle

“Great Expectations sits after ingestion and before feature engineering or training. Data enters the pipeline, GX checks it, and only a successful batch continues. The same gate can also run in CI or on scheduled data pipelines.”

## Slide 4 — Core concepts

“The Data Context manages GX objects and results. A Data Source connects to pandas. A Data Asset names our logical dataset. A Batch Definition identifies the data for one run. Expectations describe individual rules, and an Expectation Suite groups them. A Validation Definition links the batch and suite.”

## Slide 5 — Demo data contract

“Our example uses loan applications. The contract checks the schema, IDs, missing values, numeric ranges, and the target label. These are simple checks, but they cover several common failure modes that can silently damage an ML experiment.”

## Slide 6 — What happens on failure

“The bad file contains six deliberate problems. GX returns structured results with the failed expectation, affected column, count, and sample bad values. Our pipeline saves those results, prints a concise summary, returns a nonzero exit code, and never calls model training.”

## Slide 7 — What happens on success

“The clean file passes all thirteen expectations. Only then does the script split the data and fit logistic regression. The model metric is not the main point. The important point is that every training run starts from data that satisfied the same explicit contract.”

## Slide 8 — Trade-offs and production use

“GX provides readable rules, structured evidence, and integrations for dataframes, files, SQL, and Spark. The trade-off is maintenance: teams must own the expectation thresholds, control false alarms, and update contracts when the data legitimately changes. Checkpoints and actions can connect validation to automated workflows.”

## Slide 9 — Live demo

“I will first run the bad dataset and show that training is blocked. Then I will run the clean dataset and show that validation passes and metrics are written. If the live run fails for environmental reasons, the next slide contains captured output from both runs.”

## Slide 10 — Takeaway and Q&A

“Great Expectations turns assumptions about training data into executable tests. In this demo, the validation result controls whether the model can train. That moves data quality from an informal checklist into the ML pipeline.”

## Likely Q&A

**How is this different from unit tests?**  
Unit tests usually check code behavior with controlled inputs. GX evaluates properties of the actual data batch and produces structured validation results.

**Does GX clean the data?**  
No. It detects and reports violations. The pipeline decides whether to block, quarantine, alert, or trigger another action.

**Can it work with large datasets?**  
Yes. GX supports pandas, Spark, SQL databases, and file-based data sources. The execution engine pushes or distributes work according to the backend.

**How should thresholds be chosen?**  
Start from domain constraints and known failure risks. Add observed statistical ranges carefully, then review false positives and legitimate distribution changes.

**Why use an ephemeral Data Context here?**  
The demo is self-contained and recreates configuration every run. A production project usually uses a persistent context so suites, validation definitions, checkpoints, and results can be reused.
