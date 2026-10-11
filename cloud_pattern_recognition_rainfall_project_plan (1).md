# Cloud Pattern Recognition from Sky Images for Rainfall Estimation

**Project type:** Machine Vision mini project  
**Implementation style:** Agentic coding with milestone-based development  
**Recommended MVP stack:** Python, PyTorch, torchvision, OpenCV/Pillow, scikit-learn, Streamlit, pytest  
**Primary output:** Cloud-pattern classification from a sky image, plus a rainfall-risk / rainfall-amount estimate only when a suitably labelled paired dataset is available.

---

## 1. Project overview

### 1.1 Idea in simple terms

The application accepts a photograph of the sky and analyses the visible cloud patterns using computer vision. First, a deep-learning image classifier identifies the broad sky/cloud pattern. A separate rainfall model can then use the image features—and, optionally, a short sequence of recent sky images—to estimate the chance or category of rainfall in a defined future time window.

The interface should show:

1. The uploaded sky image and image-quality checks.
2. The predicted cloud/sky category and class probabilities.
3. A rainfall prediction for a clearly stated time window, such as the **next hour**, if the rainfall model has been trained and evaluated on paired image–rainfall observations.
4. A confidence/uncertainty message and a short explanation of the result.
5. The model version and a disclaimer that this is a prototype, not an official weather forecast.

### 1.2 The important scientific limitation

A cloud-image dataset usually has cloud-type labels, not measured rainfall labels. Therefore, training a cloud classifier does **not** automatically produce a valid rainfall estimator. Rainfall depends on more than the appearance of one image: cloud development over time, atmospheric moisture, wind, pressure, local terrain, and other meteorological conditions matter.

This project must keep two tasks separate:

- **Task A — Cloud-pattern recognition:** feasible using a public ground-based cloud-image dataset.
- **Task B — Rainfall estimation/nowcasting:** requires images aligned with rainfall observations for the same location and timestamps. A rain gauge is best for local ground truth. Historical weather/reanalysis data can be used for a prototype label source, but the resulting model must be described as predicting that data source's proxy labels—not as a validated local rain-gauge model.

Do not display invented rainfall values or claim real-world accuracy before this second task has been trained and tested on appropriate labels.

### 1.3 Recommended final scope

Build a working application with two clearly labelled modes:

**Mode 1: Cloud Analysis (baseline/MVP)**
- Accept JPG, JPEG, and PNG uploads.
- Validate that the file is a readable image and warn on very small, dark, corrupted, or non-sky images where feasible.
- Predict one of the cloud categories represented in the training dataset.
- Display probabilities, a bar chart, and the model's top prediction.

**Mode 2: Rainfall Estimation (data-dependent)**
- Use the image-only model as the main experiment.
- Predict rainfall category and, if the training labels and regression performance justify it, rainfall amount for the next hour.
- Keep an optional multimodal experiment (image + weather variables available at prediction time) as a comparison, not as a silent replacement for the image model.
- If no reliable paired dataset is available, show this mode as **research/demo unavailable** rather than generating a made-up estimate. A limited proxy-label demo can be enabled only when the UI explicitly says so.

This scope gives the mini project a deliverable even if paired rainfall data collection takes longer than expected.

---

## 2. Goals, questions, and non-goals

### Goals

- Build an end-to-end machine-vision application that accepts sky photographs.
- Train and evaluate a cloud-image classifier with transfer learning.
- Build a reproducible dataset and training pipeline.
- Add rainfall prediction only after establishing timestamp-matched labels.
- Evaluate both tasks with metrics appropriate to classification and regression.
- Provide an interface, tests, setup instructions, model-card notes, and a reproducible demonstration.

### Research questions

1. Can a lightweight pretrained CNN classify broad sky/cloud patterns from ground-based images?
2. How well does the cloud classifier generalize to photos taken with a different camera, location, exposure, or sky-camera geometry?
3. Can sky-image features predict rainfall in the following hour on a paired dataset, and how does performance compare with a simple baseline?
4. Does adding weather context improve performance over the image-only model? This is an optional ablation, and all input features must be available at the time of prediction.

### Non-goals for the first version

- Official meteorological warnings or safety-critical decisions.
- Guaranteed prediction of exact rainfall in millimetres from a single arbitrary photo.
- Satellite precipitation retrieval, radar processing, or numerical weather prediction from scratch.
- Claims of nationwide/global performance based only on a small or geographically narrow dataset.
- Automatic model retraining from user uploads without explicit labels and review.

---

## 3. Data strategy

### 3.1 Cloud-image datasets to inspect

Choose the dataset only after reviewing its download procedure, license, class definitions, image geometry, and available split. Do not assume that cloud labels include rainfall labels.

| Dataset/resource | Potential use | Important limitation |
|---|---|---|
| [TJNU Ground-based Cloud Dataset (GCD)](https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset) | Preferred candidate for a larger cloud-classification experiment. The repository describes 19,000 images, seven grouped cloud classes, and a 10,000/9,000 train/test split. | Read and comply with the dataset agreement before downloading or redistributing. These are cloud-classification labels, not automatically rainfall measurements. |
| [SWIMCAT dataset page](https://vintage.winklerbros.net/swimcat.html) | Small, convenient baseline for five broad sky types: clear sky, patterned clouds, thick dark clouds, thick white clouds, and veil clouds. | Only 784 images; small and geographically specific. Use as a baseline or external test, not as proof of broad generalization. Access may require following the page's request procedure. |
| [Ground-based cloud classification research using MGCD](https://pmc.ncbi.nlm.nih.gov/articles/PMC10180870/) | Research reference for a multi-modal cloud dataset. The cited study describes 8,000 images with temperature, humidity, pressure, and wind information and seven cloud categories. | Verify the actual data access and license yourself. The listed meteorological features are not necessarily ground-truth precipitation targets. |

**Recommended path:** start with GCD if access and terms allow. If its access process blocks progress, implement the initial pipeline with SWIMCAT and keep the dataset loader configurable. Never copy a dataset into a public repository unless its terms permit redistribution.

### 3.2 Rainfall target labels

For a real rainfall estimator, construct a paired table containing at minimum:

- `image_path`
- `timestamp` with timezone clearly defined
- `latitude` and `longitude` (or a stable site identifier)
- `rain_mm_next_hour` or a clearly defined observed rainfall interval
- `rain_source` (rain gauge, official station, or reanalysis/API proxy)
- `label_quality` or a documented validity flag
- optional cloud label and capture-device/site metadata

**Best target:** measured rainfall from a rain gauge or a reliable nearby meteorological station, aligned to the image capture time.

**Prototype target option:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) provides hourly historical precipitation and rain variables. It can help create a prototype table for a specified location and time range. These are model/reanalysis-based weather data, not necessarily a local gauge measurement. Record this limitation in the dataset documentation and in the user interface.

**Optional live comparison:** [Open-Meteo Forecast API documentation](https://open-meteo.com/en/docs) describes hourly forecast variables such as precipitation and precipitation probability. If shown in the application, display them as a separate external-weather comparison. Do not confuse API forecasts with the model's image-only prediction.

### 3.3 Data collection plan for paired sky images and rain

If a local camera can be used:

1. Mount a camera with a reasonably fixed view of the sky. Prefer an all-sky/wide-angle view if one is available, but document the camera geometry.
2. Capture an image every 5–15 minutes during daylight, with timestamps in a consistent timezone. Store the original image and capture metadata.
3. Measure rainfall using a rain gauge or obtain the closest credible station observation. Record the measurement interval, gauge location, units, and missing periods.
4. Match each image at time `t` to the rainfall amount during the following interval `[t, t+1 hour)`. Define this convention in the code and report so labels cannot be shifted silently.
5. Mark missing or questionable intervals as invalid; do not turn missing rainfall into zero rainfall.
6. Keep sequences from the same event/site together when splitting data to reduce leakage.

If no camera or gauge is available, the project can still deliver cloud classification plus a rainfall-model data pipeline. A proxy-labelled experiment can be included, but its result should be described accurately.

### 3.4 Rainfall categories

For an initial demonstrator, define project-specific bins on **one-hour accumulated rainfall** and make the thresholds editable in one configuration file. For example:

- `no_measurable_rain`: less than 0.1 mm in the hour
- `light`: 0.1 mm to less than 2.5 mm
- `moderate`: 2.5 mm to less than 7.5 mm
- `heavy`: 7.5 mm or more

These are **prototype bins for this project**, not a claim that they are the applicable official standard for every region. Confirm suitable local/official definitions before using them in a formal research paper or operational context. Keep the numeric regression target in millimetres even if the UI also displays categories.

### 3.5 Data-quality and leakage rules

- Split data by date, storm/event, or capture session—not merely by randomly shuffling near-duplicate images.
- Keep the test set untouched until final evaluation.
- Never use future rainfall or future weather observations as model input for a prediction at time `t`.
- If the target is next-hour rain, clearly define the input cutoff at image timestamp `t`.
- Remove corrupt files and record their count; don't silently ignore them.
- Check class balance and report samples per class before training.
- Fit normalization or any data-derived preprocessing on the training set only.
- Keep a manifest of samples and a fixed random seed for reproducibility.
- Do not report an overall accuracy alone. Rainfall data are often dominated by dry intervals, so a model that predicts “no rain” may appear deceptively strong.

---

## 4. Proposed technical architecture

### 4.1 Main workflow

```text
Sky Photo Upload
       |
       v
Image Validation and Preprocessing
       |
       v
Cloud CNN (transfer learning)
       |
       +-----------------------> Cloud class + class probabilities
       |
       v
Image Feature Representation
       |
       v
Rainfall Model (only if trained on paired image-rain labels)
       |
       +-----------------------> Next-hour rain class / amount + uncertainty
       |
       v
Streamlit Results Dashboard
```

An optional comparison branch may combine image features with weather variables that were available at prediction time. Label it separately as a multimodal model.

### 4.2 Suggested tools

| Layer | Tool | Reason |
|---|---|---|
| Language | Python 3.11 (or another version supported by the selected PyTorch release) | One language for vision, ML, API, and app code. |
| Image I/O | Pillow and/or OpenCV | Decode, resize, inspect and visualize images. |
| Deep learning | PyTorch + torchvision | Transfer learning with lightweight pretrained CNNs. |
| Baseline ML | scikit-learn | Simple baselines, metrics, splits and preprocessing utilities. |
| Data handling | pandas, NumPy | Dataset manifests and aligned time-series tables. |
| Visualization | Matplotlib | Training curves, confusion matrix and result plots. |
| User interface | Streamlit | Fast, demo-friendly upload and results interface. |
| Testing | pytest | Unit and integration tests. |
| Quality checks | Ruff (optional) | Linting and style checks. |
| Optional weather data | Open-Meteo HTTP API | Historical proxy labels or a separately displayed weather comparison. |

Keep the first release as a single Python/Streamlit application. Add FastAPI or a separate frontend only if there is a concrete deployment or API requirement.

### 4.3 Model design

**Cloud classifier baseline**

- Start with `MobileNetV3-Small` or `ResNet18` pretrained on ImageNet.
- Replace the classification head with the number of classes in the selected cloud dataset.
- Resize images to a consistent input size, initially 224×224, unless the dataset or model needs another size.
- Normalize using the pretrained model's expected normalization.
- Use conservative augmentation: horizontal/vertical flips only if they preserve the meaning of the dataset, small crops/resizing, and modest brightness/contrast changes. Do not apply aggressive augmentations that create unrealistic sky appearance.
- Begin by freezing the backbone and training the head; optionally unfreeze some final layers if validation performance supports it.
- Save the best model according to validation macro-F1, not only training accuracy.

**Rainfall estimator baseline**

A rainfall estimator is a separate model and requires paired image–rainfall labels. Recommended experiment order:

1. Simple baseline using a constant/median predictor for rainfall amount and a majority-class classifier for rain categories.
2. Image-embedding baseline: extract features from the trained CNN, then train a Random Forest or gradient-boosting/scikit-learn model. This is often easier to inspect and iterate on than a second end-to-end network.
3. Optional end-to-end model or short temporal model when enough properly aligned sequences exist.
4. Optional multimodal model using weather features available at prediction time, evaluated against the image-only model on exactly the same test examples.

For numeric prediction, consider a non-negative target transformation or a two-stage design (rain/no-rain classifier followed by amount regression on rainy periods). Report this design clearly. Clip a negative regression output to zero only as a documented post-processing rule, and evaluate its effect.

**Do not generate rainfall predictions using hand-written cloud rules and describe them as machine-learned outputs.** A heuristic may be included as an explicitly labelled baseline, but it must not be presented as a trained model.

---

## 5. Application features and screens

### Screen 1 — Home / Project Overview
- Project title and short description.
- Separate explanation for cloud classification versus rainfall estimation.
- Dataset/model status: trained, unavailable, or proxy-label experimental.
- A clear research-prototype disclaimer.

### Screen 2 — Analyze Sky Image
- Upload image control.
- Preview original image and processed image.
- Display image dimensions and validation warnings.
- Run prediction button and friendly error messages.

### Screen 3 — Cloud Analysis Results
- Top predicted cloud/sky class.
- Probability for every class in a sorted bar chart.
- Plain-language explanation derived from the class label, not fabricated reasoning.
- Optional Grad-CAM/heatmap only after it is implemented and sanity-checked; mark it as an approximate visualization, not a causal explanation.

### Screen 4 — Rainfall Estimate
- Next-hour rainfall category and predicted amount (mm), if a valid model is available.
- Prediction horizon and units displayed next to the number.
- Confidence/uncertainty indicator only if calibrated or clearly described as a model score.
- Rainfall model type: `image-only`, `multimodal`, or `proxy-label experiment`.
- When no valid rainfall model exists, show an honest unavailable message and explain what data are needed.

### Screen 5 — Model Evaluation (developer/demo mode)
- Test-set metrics and confusion matrix for cloud classification.
- Rainfall classification metrics and/or regression errors only when a rainfall test set exists.
- Dataset counts, split strategy, model version and evaluation date.
- Do not show training-set scores as final performance.

### Screen 6 — About / Limitations
- Data sources and citations.
- Class definitions and rainfall-bin definitions.
- Limitations, generalization risks and privacy note.

---

## 6. Repository structure

Use a simple, modular structure rather than putting all implementation into one file.

```text
cloud-rainfall-estimator/
├── README.md
├── AGENTS.md
├── pyproject.toml                 # or requirements.txt; choose one dependency workflow
├── .gitignore
├── .env.example                   # only if configuration/secrets are needed
├── configs/
│   ├── cloud_model.yaml
│   ├── rainfall_model.yaml
│   └── paths.yaml
├── data/
│   ├── README.md                  # download/access instructions; no restricted data committed
│   ├── raw/                       # ignored by Git
│   ├── processed/                 # ignored by Git
│   └── manifests/                 # small metadata files where permitted
├── artifacts/
│   ├── README.md                  # model artifact instructions
│   └── .gitkeep
├── notebooks/
│   └── exploratory_analysis.ipynb # optional, never the only source of pipeline logic
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_cloud_dataset.py
│   ├── build_rainfall_pairs.py
│   ├── train_cloud_classifier.py
│   ├── train_rainfall_model.py
│   └── evaluate_models.py
├── src/
│   └── cloud_rainfall/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── datasets.py
│       │   ├── manifests.py
│       │   └── validation.py
│       ├── models/
│       │   ├── cloud_classifier.py
│       │   ├── rainfall_estimator.py
│       │   └── inference.py
│       ├── services/
│       │   ├── weather_client.py
│       │   └── prediction_service.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── plots.py
│       └── ui/
│           └── app.py
├── tests/
│   ├── test_image_validation.py
│   ├── test_manifest_alignment.py
│   ├── test_inference.py
│   ├── test_rainfall_labels.py
│   └── test_ui_helpers.py
└── reports/
    ├── experiment_log.md
    ├── model_card.md
    └── figures/
```

Keep training and inference code separate. Use small functions with clear inputs/outputs. Avoid loading model weights repeatedly on every Streamlit rerun; cache the loaded model appropriately.

---

## 7. Development phases and acceptance criteria

### Phase 0 — Repository and agent rules

**Tasks**
- Create the Git repository and the structure above.
- Add `AGENTS.md` with the project constraints and expected workflow.
- Set up dependency management, formatting/linting, and pytest.
- Add an initial README with scope and limitations.

**Acceptance criteria**
- A clean environment can install dependencies using documented commands.
- `pytest` runs successfully, even if only starter tests exist.
- Raw data, credentials, caches, and large model weights are excluded from Git.

### Phase 1 — Dataset inspection and cloud-data pipeline

**Tasks**
- Document how to obtain the selected dataset and its license terms.
- Inspect class folders, file types, image sizes, duplicate images and class counts.
- Create a manifest with image paths, labels and split membership.
- Validate every image; write a report for invalid files and class imbalance.
- Make preprocessing deterministic and configurable.

**Acceptance criteria**
- The dataset inspector outputs total count, per-class counts, image-size summary and invalid-file count.
- The manifest has no missing labels or broken paths.
- The split is reproducible and respects the source dataset's official split where required.
- No dataset content is committed to Git by accident.

### Phase 2 — Cloud-classification baseline

**Tasks**
- Implement a PyTorch `Dataset` and data loaders.
- Train a majority-class baseline and a transfer-learning CNN.
- Log train/validation loss and metrics per epoch.
- Save the best checkpoint and a class-name mapping.
- Implement an inference function for one image.

**Acceptance criteria**
- Training runs on CPU for a small smoke-test subset; GPU use is optional.
- Checkpoints include model weights, class order, input size and relevant configuration.
- Inference returns a valid class and normalized class probabilities.
- The best checkpoint is selected by a validation metric, not by test-set results.

### Phase 3 — Cloud evaluation and error analysis

**Tasks**
- Evaluate the frozen final checkpoint on the held-out test split.
- Generate confusion matrix, per-class precision/recall/F1, macro-F1 and accuracy.
- Inspect examples of correct and incorrect predictions.
- Record limitations such as class imbalance, camera viewpoint, nighttime images and location bias.

**Acceptance criteria**
- Evaluation can be rerun from a documented command.
- Figures and metric tables are generated from held-out predictions.
- The report identifies the most commonly confused classes.
- No fabricated metrics or unverified performance claims appear in the README or presentation.

### Phase 4 — Paired rainfall dataset pipeline

**Tasks**
- Decide and document the rainfall target source: preferred rain gauge/station, or clearly identified API/reanalysis proxy.
- Define the timestamp, timezone, site coordinates and rainfall interval.
- Build a manifest pairing each input image with the next-hour rainfall target.
- Validate date alignment and report missing/invalid targets.
- Create date/event-aware train, validation and test splits.
- Define the project rainfall bins in configuration.

**Acceptance criteria**
- Every accepted image has one valid and unambiguous target interval.
- Tests prove that the target interval follows the image timestamp and no future label is included as a feature.
- Missing rainfall is never silently treated as zero.
- The report states whether labels are gauge observations, station observations or proxy data.

**Fallback:** if paired data are unavailable, finish a reusable pairing pipeline and skip rainfall model training. Do not bypass this phase by attaching random or fabricated rainfall labels.

### Phase 5 — Rainfall prediction experiments

**Tasks**
- Implement simple baseline models first.
- Build CNN embeddings for the paired images.
- Train a rainfall-category classifier and/or a non-negative rainfall regression model.
- Optionally evaluate a multimodal model using only prediction-time weather variables.
- Run ablations: baseline versus image-only; image-only versus multimodal if applicable.
- Check class imbalance and separate dry-period from rainy-period performance.

**Acceptance criteria**
- Use the same date-aware test set for comparable model variants.
- For classification, report macro-F1, balanced accuracy, per-class metrics and a confusion matrix; report rain/no-rain precision-recall metrics where appropriate.
- For regression, report MAE and RMSE in millimetres, with rainy-period results in addition to all-period results. R² may be included but should not be used alone.
- Compare against simple baselines.
- Persist the target definition, training configuration, feature list, label source and split identifiers alongside model artifacts.
- If results do not beat the baseline or do not generalize, present that result honestly.

### Phase 6 — Inference service and Streamlit interface

**Tasks**
- Implement upload, validation, preprocessing and model loading.
- Display image preview, cloud predictions and probabilities.
- Enable rainfall results only if a compatible rainfall checkpoint and metadata are present.
- Add clear errors for missing model files, unsupported formats and invalid images.
- Add a model/dataset information panel and disclaimer.

**Acceptance criteria**
- The app launches with the documented command.
- A valid image can be analysed end-to-end.
- Invalid inputs produce an understandable message rather than a traceback.
- Cloud and rainfall outputs are clearly separated.
- The app never shows a fabricated rainfall estimate when the rainfall model is absent.

### Phase 7 — Testing, robustness and reproducibility

**Tasks**
- Unit-test image validation, preprocessing, class mapping, rainfall-bin boundaries and timestamp pairing.
- Smoke-test model loading and inference on sample images.
- Test CPU inference and missing-artifact behavior.
- Add checks for malformed images and unexpected image dimensions.
- Verify that the model artifact and config agree on class order and input size.

**Acceptance criteria**
- All tests pass from a clean environment.
- Training, evaluation and app commands are documented and reproducible.
- Random seeds and software versions are recorded.
- Secrets and private data are not logged or committed.

### Phase 8 — Final report and demo packaging

**Tasks**
- Create `reports/model_card.md` documenting intended use, data, metrics, caveats and known failure modes.
- Update README with setup, data access, training, evaluation, demo, screenshots and troubleshooting.
- Add architecture and data-flow diagrams.
- Prepare a short demo script and a results table generated from the actual test run.
- List unfinished goals openly, especially if rainfall labels remain proxy-based or unavailable.

**Acceptance criteria**
- A reviewer can reproduce the cloud model evaluation and launch the app using the README.
- The report distinguishes completed functions from planned or experimental functions.
- Dataset citations and relevant usage restrictions are included.
- All numbers in the report match generated artifacts.

---

## 8. Practical 14-day schedule

This is a planning estimate, not a guarantee; obtaining paired rainfall data is the critical dependency.

| Day | Deliverable |
|---|---|
| 1 | Repository, environment, `AGENTS.md`, README skeleton |
| 2 | Dataset access, license review, dataset inspector |
| 3 | Clean manifest, train/validation/test strategy, sample visualization |
| 4 | Majority baseline and CNN data pipeline |
| 5 | First cloud-model training run |
| 6 | Hyperparameter refinement and checkpoint saving |
| 7 | Held-out evaluation, confusion matrix, error analysis |
| 8 | Decide paired rain-label source and implement timestamp alignment |
| 9 | Validate paired samples, leakage checks, rain-bin configuration |
| 10 | Rainfall baseline; skip training if paired data are inadequate |
| 11 | Image-embedding rain experiment and metrics, if the data support it |
| 12 | Streamlit upload, cloud result view and honest rainfall-mode gating |
| 13 | Tests, model-card, robustness fixes and documentation |
| 14 | Final demo, report, limitations and repository cleanup |

If rainfall labels are not available by Day 8, focus Days 8–11 on the pairing pipeline, uncertainty/limitation handling, and cloud model evaluation rather than pretending that a rainfall model is trained.

---

## 9. Agentic coding workflow

Use an agentic coding tool such as Cursor, Claude Code, or GitHub Copilot, but operate it in small, reviewable tasks. Give it this file as the source of truth.

### 9.1 Rules for the coding agent

Add these instructions to `AGENTS.md`:

```text
Project: Cloud Pattern Recognition from Sky Images for Rainfall Estimation.

1. Read README.md and this project plan before editing code.
2. Work on one milestone at a time. Before changing files, state the files to create/edit and the acceptance criteria being addressed.
3. Do not invent datasets, labels, results, metrics, API responses, or trained model files.
4. Never train or claim a rainfall predictor without timestamp-paired image and rainfall labels. Keep proxy/reanalysis labels explicitly identified.
5. Keep cloud classification and rainfall prediction as separate tasks and separate evaluation reports.
6. Do not use test-set results for model selection or preprocessing decisions.
7. Prevent time leakage: a prediction made at image time t cannot use rainfall observations after t as input features.
8. Put reusable logic in src/cloud_rainfall/; scripts should call that logic rather than duplicating it.
9. Add or update pytest tests for every data-processing or model-inference feature.
10. Run the relevant tests and report the actual command and result. Never claim tests passed unless they were run.
11. Keep raw datasets, large checkpoints, API keys, .env files and user uploads out of Git unless explicitly approved.
12. If a requirement cannot be implemented due to missing data or credentials, explain the blocker and implement a safe stub or documented fallback rather than fabricating output.
13. Keep dependency choices minimal. Do not add a service, database, or framework without a clear need.
14. Update README.md and reports/experiment_log.md when user-visible behavior or experiment results change.
```

### 9.2 Recommended prompt sequence

Run these prompts in order. Do not ask the coding agent to build the entire system in one pass.

**Prompt 1 — Repository foundation**

```text
Read the project plan and create only Phase 0. Set up the Python project, the modular folder structure, dependency configuration, .gitignore, a starter README.md, AGENTS.md, and a minimal pytest configuration. Do not implement training or invent model files. Run the tests and report the exact commands/results. Keep the dependency list minimal.
```

**Prompt 2 — Dataset inspection**

```text
Implement Phase 1 only. Build a configurable dataset inspector and manifest creator for a folder-based cloud-image dataset. Report class counts, image dimensions, corrupt/unreadable files, duplicate hashes, and manifest validation. Respect any official dataset split and do not move or redistribute source data. Add tests using temporary synthetic images and a small fixture dataset. Do not start model training yet.
```

**Prompt 3 — Cloud classifier**

```text
Implement Phase 2 using PyTorch and torchvision transfer learning. Start with MobileNetV3-Small unless there is a documented compatibility issue. Use the manifest and configurable class mapping; save the best validation checkpoint with model configuration and class names. Include a small CPU smoke-test mode. Add tests for output dimensions, probability normalization, and checkpoint metadata. Do not tune against the test split and do not claim performance before evaluation.
```

**Prompt 4 — Evaluation**

```text
Implement Phase 3. Add a separate evaluation command that loads the selected checkpoint and evaluates only the held-out test split. Generate accuracy, per-class precision/recall/F1, macro-F1, confusion matrix, and a CSV of predictions. Record the split strategy and random seed. Make every report reproducible from a command. Do not fabricate or hard-code metric values.
```

**Prompt 5 — Rainfall-pairing pipeline**

```text
Implement Phase 4 only. Design a manifest that aligns an image captured at timestamp t with measured or explicitly proxy-labelled precipitation during [t, t+1 hour). Include timezone handling, missing-value handling, configurable rainfall bins, and validation reports. Never convert missing labels to zero. Add tests for boundary timestamps, daylight-saving/timezone parsing where relevant, missing intervals, and prevention of feature/target leakage. Do not train a rainfall model until the pairing report is valid.
```

**Prompt 6 — Rainfall baseline and model**

```text
Inspect the paired dataset and its label source before implementing Phase 5. If the number, quality, diversity, or alignment of paired samples is inadequate, stop and report why rather than fabricating data. Otherwise implement simple baselines, image embeddings from the saved CNN, and a suitable rainfall classifier/regressor. Use date/event-aware splits, compare against baseline models, and report macro-F1/balanced accuracy for classification and MAE/RMSE in mm for regression. Save the label source, target definition, features and split metadata with each artifact.
```

**Prompt 7 — Streamlit app**

```text
Implement Phase 6 as a clean Streamlit application. Add image upload, preview, input validation, cloud-class probabilities, and clearly separated rainfall results. Cache model loading. If a rainfall checkpoint or metadata is missing, show an honest 'Rainfall model not available' state; never use random values or hand-written heuristics as if they were model predictions. Handle file/model errors gracefully. Add helper-function tests and update README run instructions.
```

**Prompt 8 — Final verification**

```text
Run Phase 7 and Phase 8. Review the project plan acceptance criteria one by one. Run all tests, inspect Git status for data/secrets/large artifacts, confirm commands from README work, and create reports/model_card.md and reports/experiment_log.md using only actual results. List any unmet criteria and blockers. Do not report success for anything that was not tested.
```

### 9.3 How to review the agent's work

After every prompt:

1. Inspect the changed files before accepting the changes.
2. Ask for the exact test/build command and its actual output summary.
3. Run the command yourself when practical.
4. Check the Git diff for accidental data files, secrets, duplicated code, hard-coded paths and fake metrics.
5. Commit the milestone only when its acceptance criteria pass.

Suggested commit sequence:

```text
chore: establish project scaffold
feat(data): add cloud dataset manifest pipeline
feat(vision): add cloud transfer-learning baseline
feat(eval): add cloud model evaluation reports
feat(data): align sky images with rainfall observations
feat(rain): add labelled rainfall baseline experiments
feat(ui): add Streamlit prediction interface
test: add end-to-end smoke tests
docs: finalize model card and reproducibility guide
```

---

## 10. Metrics and result reporting

### Cloud classification

Report:
- Test accuracy.
- Macro-F1 and weighted-F1.
- Per-class precision, recall, and F1.
- Confusion matrix.
- Number of samples per class and split.
- Training settings, pretrained backbone, image size, optimizer, learning rate, batch size, epochs, seed, and software versions.

Macro-F1 is important when cloud classes are imbalanced. A high accuracy alone can hide poor performance for less frequent classes.

### Rainfall classification

Report:
- Macro-F1 and balanced accuracy.
- Per-class precision, recall and F1.
- Confusion matrix.
- Rain/no-rain precision, recall and PR-AUC where useful.
- Performance by label source and location when the sample size supports it.

### Rainfall regression

Report:
- MAE in millimetres.
- RMSE in millimetres.
- Optionally R², with caveats.
- All-interval performance and rainy-interval performance separately.
- A scatter plot of observed versus predicted rainfall and a residual/error plot.
- The proportion of dry intervals in the evaluation set.

Always compare with a simple baseline, such as a majority-class classifier and a zero/median rainfall predictor. If the model does not outperform the baseline, say so.

---

## 11. Deployment and reproducibility

For a mini project, local deployment is sufficient:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt   # only if requirements.txt is the chosen dependency workflow
python scripts/inspect_dataset.py --help
streamlit run src/cloud_rainfall/ui/app.py
```

Adapt commands to the repository's actual dependency and CLI configuration. If using `pyproject.toml` only, document the install command for that setup instead of maintaining two competing dependency files.

Before submission:
- Pin or record package versions.
- Record the Python version and training device.
- Keep large model files out of Git if repository limits apply; document how to obtain them.
- Provide a tiny openly licensed sample only when permitted, otherwise use a user-supplied sample or a synthetic test fixture.
- Never commit API keys. A public no-key weather endpoint should still be called with timeouts, error handling and appropriate usage/attribution notes.
- Do not make the user upload images to an external service unless this behavior is clearly disclosed.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cloud datasets have no rainfall labels | Treat cloud classification as Task A. Build a separate paired rainfall dataset before training Task B. |
| Rain API or reanalysis does not match local rainfall | Label it as proxy data; prefer a nearby rain gauge for validation. |
| Images from one camera/location dominate | Use separate dates/events/sites for validation and test; document the domain. |
| Random split leaks near-identical frames | Split by capture session, day, storm or site rather than by image alone. |
| Many dry hours inflate metrics | Report rainy-only regression results, balanced metrics and simple baselines. |
| Night, glare, buildings or poor framing confuse the model | Validate image quality, document supported conditions and test failure cases. |
| Agent produces code that appears complete but is untested | Require milestone acceptance criteria, tests and review of actual command output. |
| Restricted dataset accidentally committed | Read license, keep data outside Git, add ignore rules, and check `git status` before every commit. |
| User interprets prototype as an official forecast | Display horizon, target definition, label source, limits and prototype disclaimer in the app. |

---

## 13. Final deliverables checklist

- [ ] Source repository with modular Python code.
- [ ] `AGENTS.md` containing agentic coding rules.
- [ ] Documented data source, license/access steps and preprocessing pipeline.
- [ ] Cloud dataset manifest and data-quality report.
- [ ] Trained cloud classifier checkpoint and class mapping.
- [ ] Held-out cloud evaluation metrics, confusion matrix and error analysis.
- [ ] Paired rainfall dataset manifest, or explicit documentation that it is not yet available.
- [ ] Rainfall model and held-out evaluation only if valid paired labels exist.
- [ ] Streamlit application with upload validation and clear output labels.
- [ ] Tests for preprocessing, inference, rainfall bins and time alignment.
- [ ] README setup, training, evaluation and run instructions.
- [ ] Model card documenting intended use, data source, results and failure modes.
- [ ] Final demo script and report that use actual measured results.

---

## 14. Recommended execution order

Start with **Phase 0 → Phase 1 → Phase 2 → Phase 3**. This secures a working image-classification project first. Then determine whether paired rainfall labels can be collected or accessed. Implement Phases 4 and 5 only when that data is available and valid; then connect whichever trained models actually exist to the Streamlit app.

The strongest defensible mini-project result is not a dashboard that always shows a rainfall number. It is a reproducible machine-vision pipeline that classifies sky patterns, tests a rainfall-estimation hypothesis against time-aligned rainfall labels, compares with baselines, and clearly communicates its limitations.

## 15. Reference links

1. Liu et al., *Ground-based Remote Sensing Cloud Classification via Context Graph Attention Network* (2022), and the associated [TJNU Ground-based Cloud Dataset repository](https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset).
2. [SWIMCAT dataset page](https://vintage.winklerbros.net/swimcat.html) and the associated dataset-description links on that page.
3. [MMST: A Multi-Modal Ground-Based Cloud Image Classification Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC10180870/) for an example of cloud images paired with meteorological variables.
4. [Open-Meteo Historical Weather API documentation](https://open-meteo.com/en/docs/historical-weather-api) for historical hourly precipitation variables and data interpretation.
5. [Open-Meteo Forecast API documentation](https://open-meteo.com/en/docs) for forecast precipitation variables and hourly output structure.
6. [Open-source ground-based sky image dataset survey](https://arxiv.org/abs/2211.14709) for further dataset discovery and dataset-selection considerations.
