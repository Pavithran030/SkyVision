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
                                                │ FastAPI Web Dashboard    │
                                                └──────────────────────────┘
```

---

## 📂 3. Professional Project Structure (Zero Deep Nesting)

```text
Mini_Project/
├── app.py                      # FastAPI Web Application & API Server (http://127.0.0.1:8000)
├── config.py                   # Central paths, hyper-parameters, and domain definitions
├── train.py                    # Unified training pipeline (Task A Vision + Task B Nowcasting)
├── evaluate.py                 # Evaluation benchmark on held-out test splits
├── static/                     # Modern responsive dashboard (Single-Page Application)
│   ├── index.html              # Clean Tailwind CSS & Chart.js dashboard
│   ├── app.js                  # Asynchronous client (drag & drop, live charts, API fetch)
│   └── styles.css              # Custom styles and animations
├── src/                        # Core reusable Python package (flat & modular)
│   ├── __init__.py
│   ├── dataset.py              # Validation, transforms, PyTorch Dataset, pairing
│   ├── models.py               # MobileNetV3-Small CNN + Random Forest estimators
│   ├── inference.py            # Single-image inference engine + PredictionService
│   └── metrics.py              # Accuracy, Macro-F1, confusion matrix & plot utilities
├── data/                       # Benchmark data & metadata manifests
│   ├── samples/                # Benchmark demo sky photos (5 categories)
│   ├── cloud_manifest.csv      # Cloud image paths & stratified splits
│   └── rainfall_manifest.csv   # Aligned timestamps & next-hour rainfall observations
├── artifacts/                  # Trained model checkpoints & metadata
│   ├── cloud_classifier.pt     # Best fine-tuned MobileNetV3-Small model
│   ├── rainfall_classifier.joblib  # Random Forest rain category classifier
│   ├── rainfall_regressor.joblib   # Random Forest continuous precipitation regressor
│   ├── multimodal_classifier.joblib# Multimodal (Image + Weather) model
│   └── metadata.json           # Model parameters & test metrics
├── reports/                    # Benchmark figures & evaluation logs
│   ├── confusion_matrix.png    # Test split normalized confusion matrix
│   ├── rainfall_scatter.png    # Observed vs. predicted precipitation scatter plot
│   ├── model_card.md           # Model card specifications & physical caveats
│   └── experiment_log.md       # Chronological experiment history
├── tests/                      # Automated pytest test suite
│   ├── test_api.py             # FastAPI REST endpoints & static delivery
│   ├── test_data.py            # Image validation, transforms & date-aware splits
│   ├── test_models.py          # MobileNetV3 forward pass & Random Forest training
│   └── test_end_to_end.py      # Full pipeline integration verification
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package configuration
├── .gitignore                  # Git exclusions for weights, data, and virtualenv
├── .env.example                # Example environment variables
├── AGENTS.md                   # Strict agentic engineering workflow rules
└── README.md                   # Complete documentation and viva guide
```

---

## ⚙️ 4. Local Installation & Setup

This project uses a dedicated local Python virtual environment (`.venv`). Do **not** use the global Python environment or Docker.

### Step 1: Activate the Virtual Environment
```powershell
# In Windows PowerShell inside E:\Academics\MV\Mini_Project:
.\.venv\Scripts\Activate.ps1
```

### Step 2: Run Tests to Verify Environment
```powershell
.\.venv\Scripts\pytest.exe -v
```
All **12 automated tests** will pass in seconds!

---

## 🚀 5. How to Run the Project Pipeline

Every step is executed directly from the project root using simple commands:

### 1. Launch the Web Dashboard Application
```powershell
.\.venv\Scripts\python.exe app.py
```
Open your browser at:
- **Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. (Optional) Re-train Models from Scratch
To run the full end-to-end training pipeline:
```powershell
.\.venv\Scripts\python.exe train.py --all
```
Or train individual components:
- Vision CNN only: `.\.venv\Scripts\python.exe train.py --vision`
- Rainfall Nowcasting only: `.\.venv\Scripts\python.exe train.py --rainfall`

### 3. (Optional) Re-run Test Set Evaluation Benchmark
```powershell
.\.venv\Scripts\python.exe evaluate.py
```
This generates the accuracy and Macro-F1 report and saves `reports/confusion_matrix.png`.

Interactive OpenAPI / Swagger documentation is also accessible at:
`http://127.0.0.1:8000/docs`

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

When you demo the project locally (`python scripts/run_server.py`), you can present the 3 main navigation views:

1. **Live Analysis:**  
   - **Input Section:** Drag-and-drop file upload zone or 1-click **Benchmark Demo Samples** selector (allowing instant switching between clear sky, ripple altocumulus, dark storm clouds, etc.).
   - **Real-Time Image HUD:** Image dimensions, RGB channels, and warning alerts (for underexposure or washed-out images).
   - **Interactive Weather Panel:** Dynamic sliders for Temperature, Relative Humidity, Pressure, and Wind Speed to evaluate multimodal predictions.
   - **Task A Card (Cloud Pattern Recognition):** Top predicted sky category, confidence badge, meteorological explanation, and an interactive **Chart.js animated horizontal bar chart** showing all 5 class probabilities.
   - **Task B Card (Rainfall Nowcasting):** Next-hour discrete rain category, expected precipitation amount (mm), horizon indicator (`Next 1 Hour [t, t+1h)`), and Multimodal Context comparison badge.

2. **Model Evaluation:**  
   - Held-out test accuracy benchmark cards (77.78% Test Acc, 80.00% Balanced Acc, 0.7200 Macro-F1).
   - Comprehensive comparative performance table (Baseline vs. Image-only vs. Multimodal).
   - High-resolution normalized Confusion Matrix heatmap.
   - Observed vs. Predicted rainfall regression scatter plot.

3. **Scientific Foundations & Limitations:**  
   - Detailed domain explanations of physical camera geometry, moisture columns, and proxy reanalysis disclosures.

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
