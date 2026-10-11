# AGENTS.md

## Project Constraints and Workflow Rules

Project: Cloud Pattern Recognition from Sky Images for Rainfall Estimation.

1. Read `README.md` and `cloud_pattern_recognition_rainfall_project_plan (1).md` before editing code.
2. Work on one milestone at a time. Before changing files, state the files to create/edit and the acceptance criteria being addressed.
3. Do not invent datasets, labels, results, metrics, API responses, or trained model files.
4. Never train or claim a rainfall predictor without timestamp-paired image and rainfall labels. Keep proxy/reanalysis labels explicitly identified.
5. Keep cloud classification and rainfall prediction as separate tasks and separate evaluation reports.
6. Do not use test-set results for model selection or preprocessing decisions.
7. Prevent time leakage: a prediction made at image time $t$ cannot use rainfall observations after $t$ as input features.
8. Put reusable logic in `src/cloud_rainfall/`; scripts should call that logic rather than duplicating it.
9. Add or update pytest tests for every data-processing or model-inference feature.
10. Run the relevant tests and report the actual command and result. Never claim tests passed unless they were run.
11. Keep raw datasets, large checkpoints, API keys, `.env` files and user uploads out of Git unless explicitly approved.
12. If a requirement cannot be implemented due to missing data or credentials, explain the blocker and implement a safe stub or documented fallback rather than fabricating output.
13. Keep dependency choices minimal. Do not add a service, database, or framework without a clear need.
14. Update `README.md` and `reports/experiment_log.md` when user-visible behavior or experiment results change.
