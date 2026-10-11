# ⛅ SkyVision: Cloud Pattern Recognition from Sky Images for Rainfall Estimation

**Machine Vision Mini Project**  
*Deep Transfer Learning for Sky Pattern Classification & Timestamp-Aligned Precipitation Nowcasting*

---

## 📖 1. Project Overview

This project implements an end-to-end computer vision and machine learning system that analyzes ground-based sky photographs. The core system performs two distinct, scientifically separated tasks:

1. **Task A — Cloud Pattern Recognition (Vision Task):**  
   Accepts a daylight sky photograph, verifies image quality (resolution, color channels, contrast, overexposure/darkness), and classifies the visible cloud morphology using an ImageNet pretrained **MobileNetV3-Small** convolutional neural network into 5 standard sky categories:
   - `clear_sky` (Clear blue sky gradient)
   - `patterned_clouds` (Rippled altocumulus / stratocumulus)
   - `thick_dark_clouds` (Dense dark cumulonimbus / nimbostratus)
   - `thick_white_clouds` (Bright reflective cumulus)
   - `veil_clouds` (Translucent cirrostratus sheet)

2. **Task B — Short-Term Rainfall Estimation / Nowcasting (Nowcasting Task):**  
   Uses high-dimensional latent bottleneck embeddings (1024-dim) extracted from the trained Cloud CNN to estimate:
   - Discrete rainfall category for the **next 1 hour** (`no_measurable_rain`, `light`, `moderate`, `heavy`).
   - Expected continuous accumulated precipitation in millimetres (mm).
   - *Multimodal Comparison*: Integrates visual embeddings with contemporaneous surface weather context (temperature, relative humidity, pressure, wind speed) to test whether atmospheric state improves purely optical predictions.

> [!IMPORTANT]
> **Scientific Integrity Rule**: A photograph of the sky alone cannot guarantee exact rainfall amounts because rain formation depends on vertical column moisture, atmospheric thermodynamics, and updrafts. Therefore:
> - Cloud classification and rainfall nowcasting are decoupled into separate models and separate evaluation reports.
> - Rainfall targets are strictly aligned to the next-hour observation interval $[t, t+1\text{ hour})$. No future weather variables are ever leaked into prediction inputs.
> - Models trained on reanalysis data are transparently badged as `proxy-label experiment`.

---

## 🏗️ 2. System Architecture

```text
               Ground Sky Image
                      │
                      ▼
         ┌─────────────────────────┐
         │ Image Quality & Domain  │  --> Checks dimensions (>=64x64),
         │ Validation Checks       │      color channels, washouts, darkness
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │   MobileNetV3-Small     │  --> Lightweight CNN with
         │   Transfer Learning     │      ImageNet weights
         └────────────┬────────────┘
                      │
         ┌────────────┴──────────────────────────┐
         │                                       │
         ▼                                       ▼
 ┌──────────────────────┐             ┌──────────────────────┐
 │   Task A: Vision     │             │ 1024-Dim Latent      │
 │   Softmax Head       │             │ Visual Embeddings    │
 └──────────┬───────────┘             └──────────┬───────────┘
            │                                    │
            ▼                                    ├──────────────────────────┐
   Cloud Category &                              │                          │
   Class Probabilities                           ▼                          ▼
                                       ┌──────────────────┐       ┌──────────────────┐
                                       │ Task B: Image    │       │ Multimodal Model │
                                       │ Random Forest    │       │ (Image + Weather)│
                                       └────────┬─────────┘       └────────┬─────────┘
                                                │                          │
                                                ▼                          ▼
                                        Next-Hour Rain Bin &       Enhanced Multimodal
                                        Amount in mm               Risk Assessment
                                                │                          │
                                                └────────────┬─────────────┘
                                                             │
                                                             ▼
                                                ┌──────────────────────────┐
                                                │ Streamlit Web Dashboard  │
                                                └──────────────────────────┘
```

---

## 📂 3. Repository Structure

```text
Mini_Project/
├── AGENTS.md                                # Strict agentic coding rules & constraints
├── README.md                                # Complete system documentation and guide
├── pyproject.toml                           # Package configuration & pytest settings
├── requirements.txt                         # Pinned Python package dependencies
├── .gitignore                               # Git ignore rules for data, weights, and caches
├── .env.example                             # Environment variables and API endpoints
├── configs/
│   ├── paths.yaml                           # Directory paths and artifact locations
│   ├── cloud_model.yaml                     # CNN architecture, transforms, classes
│   └── rainfall_model.yaml                  # Rain bin thresholds, weather features
├── data/
│   ├── raw/                                 # Raw sky images (excluded from Git)
│   ├── processed/                           # Intermediate caches (excluded from Git)
│   └── manifests/
│       ├── cloud_manifest.csv               # Image paths, labels, stratified splits
│       └── rainfall_manifest.csv            # Aligned timestamps, next-hour rain labels
├── artifacts/
│   ├── cloud_classifier_best.pt             # Best trained MobileNetV3 checkpoint
│   ├── rainfall_classifier_best.joblib      # Random Forest rain category classifier
│   ├── rainfall_regressor_best.joblib       # Random Forest continuous rain regressor
│   ├── multimodal_classifier_best.joblib    # Multimodal (Image + Weather) model
│   └── model_metadata.json                  # Artifact metadata & evaluation metrics
├── scripts/
│   ├── prepare_cloud_dataset.py             # Generates sample data or ingests raw images
│   ├── inspect_dataset.py                   # Data quality audit and manifest inspection
│   ├── train_cloud_classifier.py            # Fine-tunes MobileNetV3 on validation macro-F1
│   ├── evaluate_models.py                   # Evaluates checkpoint on held-out test split
│   ├── build_rainfall_pairs.py              # Aligns image timestamps with rainfall targets
│   └── train_rainfall_model.py              # Trains baseline, image, and multimodal models
├── src/
│   └── cloud_rainfall/
│       ├── config.py                        # Centralized YAML configuration loader
│       ├── data/
│       │   ├── validation.py                # PIL/NumPy image validation checks
│       │   ├── manifests.py                 # Folder scanning, hashing, stratified split
│       │   ├── datasets.py                  # PyTorch Dataset and augmentations
│       │   └── rainfall_pairing.py          # Time alignment and date-aware splitting
│       ├── models/
│       │   ├── cloud_classifier.py          # MobileNetV3-Small module & checkpoint I/O
│       │   ├── inference.py                 # Single-image inference engine
│       │   └── rainfall_estimator.py        # Random Forest & Dummy baseline models
│       ├── services/
│       │   ├── weather_client.py            # Open-Meteo historical & forecast client
│       │   └── prediction_service.py        # Coordinated inference service
│       ├── evaluation/
│       │   ├── metrics.py                   # Macro-F1, Balanced Acc, MAE, RMSE in mm
│       │   └── plots.py                     # Confusion matrix, bar chart, scatter plots
│       └── ui/
│           └── app.py                       # Full multi-screen Streamlit application
├── tests/
│   ├── test_starter.py                      # Phase 0 setup tests
│   ├── test_image_validation.py             # Image quality & corruption checks
│   ├── test_manifest_alignment.py           # Manifest creation & split integrity
│   ├── test_inference.py                    # PyTorch forward pass & probability shapes
│   ├── test_rainfall_labels.py              # Rain bin boundaries & time leakage checks
│   ├── test_ui_helpers.py                   # UI helper & service fallback tests
│   └── test_end_to_end.py                   # Complete pipeline integration test
└── reports/
    ├── experiment_log.md                    # Chronological record of all experiments
    ├── model_card.md                        # Model card specifications and caveats
    ├── cloud_test_predictions.csv           # Per-sample test split predictions
    └── figures/
        ├── cloud_confusion_matrix.png       # Test split confusion matrix plot
        └── rainfall_observed_vs_predicted.png# Observed vs. predicted scatter plot
```

---

## ⚙️ 4. Local Installation & Setup

This project uses a dedicated local Python virtual environment (`.venv`). Do **not** use the global Python environment or Docker.

### Step 1: Clone or navigate to the directory
```powershell
cd E:\Academics\MV\Mini_Project
```

### Step 2: Activate the Virtual Environment
```powershell
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```
*(If execution policies prevent activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell first).*

### Step 3: Verify the installation
```powershell
.\.venv\Scripts\pytest.exe -v
```
All **19 unit and integration tests** will run and pass!

---

## 🚀 5. How to Run the Project Pipeline

Every step is reproducible with a single command from PowerShell:

### 1. Prepare Dataset & Manifest
Generate or ingest sample sky images and create stratified train/val/test splits:
```powershell
.\.venv\Scripts\python.exe scripts/prepare_cloud_dataset.py --create-sample --samples-per-class 24
```

### 2. Inspect Dataset Quality
Inspect class balance, dimensions, and corrupt file count:
```powershell
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
```

### 3. Train the Cloud Classifier (Task A)
Fine-tunes MobileNetV3-Small on CPU using AdamW optimizer, saving the checkpoint with highest validation macro-F1:
```powershell
.\.venv\Scripts\python.exe scripts/train_cloud_classifier.py --epochs 5 --lr 0.001
```

### 4. Evaluate Cloud Classifier on Held-Out Test Split
Evaluates strictly on the held-out test split (18 samples), generating precision/recall/F1 and the confusion matrix:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate_models.py
```

### 5. Build Timestamp-Aligned Rainfall Dataset
Aligns sky image timestamps with next-hour precipitation targets using date-aware splits:
```powershell
.\.venv\Scripts\python.exe scripts/build_rainfall_pairs.py --generate-proxy-pairs
```

### 6. Train Rainfall Nowcasting Models (Task B)
Extracts 1024-dim CNN embeddings and trains the Baseline, Image-only Random Forest, and Multimodal models:
```powershell
.\.venv\Scripts\python.exe scripts/train_rainfall_model.py
```

### 7. Launch the Streamlit Web Application
Launch the interactive web dashboard:
```powershell
.\.venv\Scripts\streamlit.exe run src/cloud_rainfall/ui/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 6. Empirical Results & Performance Benchmark

All metrics reported below were computed strictly on held-out test splits without data leakage:

### Task A: Cloud Pattern Classification (MobileNetV3-Small)
- **Held-Out Test Accuracy**: **77.78%**
- **Balanced Accuracy**: **80.00%**
- **Macro-F1 Score**: **0.7200**
- **Per-Class Metrics**:
  - `clear_sky`: Precision 100%, Recall 100%, F1 = 1.0000
  - `patterned_clouds`: Precision 100%, Recall 100%, F1 = 1.0000
  - `thick_white_clouds`: Precision 100%, Recall 100%, F1 = 1.0000
  - `veil_clouds`: Precision 42.9%, Recall 100%, F1 = 0.6000
  - `thick_dark_clouds`: Precision 0.0%, Recall 0.0% (confused with dense veil overcast)

### Task B: Short-Term Rainfall Nowcasting
| Model Variant | Macro-F1 | Accuracy | MAE (mm) | RMSE (mm) | Description |
|---|---|---|---|---|---|
| **Majority / Median Baseline** | 0.1667 | 50.0% | 3.88 mm | 6.94 mm | Naive benchmark |
| **Image-Only CNN Embeddings** | **0.2559** | 40.0% | **2.83 mm** | **4.53 mm** | Random Forest on 1024-dim visual vectors |
| **Multimodal (Vision + Weather)** | **0.3145** | 40.0% | N/A | N/A | Embeddings + Temp, Humidity, Pressure, Wind |

**Key Finding**:
The deep feature representations extracted by the CNN reduced rainfall estimation Mean Absolute Error by **1.05 mm** compared to the naive baseline. Furthermore, providing contemporaneous weather context improved classification Macro-F1 from **0.2559** to **0.3145**.

---

## 🖥️ 7. Walkthrough of the Web Dashboard Screens

When you demo the project locally (`streamlit run src/cloud_rainfall/ui/app.py`), you can present the 4 main navigation sections:

1. **Home / Overview:**  
   Presents the project title, the scientific distinction between cloud classification and rainfall nowcasting, live sidebar status indicators showing whether model checkpoints are loaded, and the academic prototype disclaimer.

2. **Analyze Sky Image:**  
   - Allows uploading any sky photo (JPG, PNG, WEBP) or selecting a sample image from the dataset.
   - Shows live image metadata (dimensions, intensity) and validation warnings (e.g. darkness or overexposure).
   - An expandable weather panel allows tuning contemporaneous surface weather variables (temperature, relative humidity, pressure, wind speed).
   - Clicking **"Run Vision & Nowcasting Analysis"** displays:
     - Top predicted cloud category with confidence score.
     - Sorted probability bar chart for all 5 cloud types.
     - Meteorological description explaining the cloud morphology.
     - Next-hour rainfall category, estimated precipitation in mm, prediction horizon, and model provenance badge (`image-only` or `proxy-label experiment`).

3. **Model Evaluation:**  
   - **Tab 1 (Cloud Classification):** Shows held-out test accuracy (77.78%), the normalized confusion matrix heatmap, and a sample predictions table.
   - **Tab 2 (Rainfall Nowcasting):** Displays model comparison metrics, the scatter plot of Observed vs. Predicted rainfall (mm), and model metadata JSON.

4. **About & Scientific Limits:**  
   Provides references to SWIMCAT, TJNU-GCD, and Open-Meteo, with full discussion on domain adaptation and physical limitations.

---

## 🎓 8. Viva & Presentation Talking Points

If your evaluator or professor asks you about the project, here is how you can explain each component:

1. **"Why didn't you train an end-to-end network directly from image to rainfall?"**  
   *Answer*: Cloud photograph datasets (like SWIMCAT or TJNU-GCD) have cloud labels, not physical rain gauge readings. Training directly would fabricate rainfall without ground truth. By separating Task A (Vision) from Task B (Nowcasting), we extract rich transfer-learned visual representations (1024-dim embeddings) and align them with timestamped rainfall observations, preventing false claims.

2. **"How did you prevent data leakage?"**  
   *Answer*: In vision, we used stratified splitting. In rainfall nowcasting, standard random shuffling is dangerous because consecutive photos from the same storm look identical. We implemented **date-aware splitting**, where all photos from a given day/storm remain strictly in the same split (train, validation, or test). Furthermore, we ensure the prediction at time $t$ only targets $[t, t+1\text{ hour})$, and only uses weather inputs available at or before $t$.

3. **"Why did you choose MobileNetV3-Small over ResNet50 or larger models?"**  
   *Answer*: MobileNetV3-Small is lightweight (~9 MB checkpoint), highly efficient on CPU, and incorporates depthwise separable convolutions and squeeze-and-excitation blocks. It provides excellent feature representations without demanding high compute resources, making it ideal for local real-time edge or desktop deployment.

4. **"What metrics did you prioritize and why?"**  
   *Answer*: Accuracy alone is misleading in meteorological datasets because clear skies and dry hours dominate (class imbalance). We prioritized **Macro-F1** and **Balanced Accuracy** for classification, and **MAE in millimetres** for regression.
