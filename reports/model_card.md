# Model Card: Cloud Pattern Recognition & Short-Term Rainfall Nowcasting

## 1. Model Details
- **Model Name**: SkyVision (Cloud CNN & Rainfall Nowcaster)
- **Model Version**: 0.1.0
- **Architectures**:
  - **Vision Backbone**: MobileNetV3-Small (ImageNet pretrained weights, fine-tuned linear classification head).
  - **Feature Representation**: 1024-dimensional bottleneck latent representation.
  - **Rainfall Classifier & Regressor**: Random Forest (100 estimators, balanced class weights).
  - **Multimodal Model**: Random Forest on concatenated visual embeddings and surface meteorological variables (`temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`).
- **Input Modality**: Ground-based RGB sky photograph (normalized to 224x224 px) + optional contemporaneous surface weather features.
- **Output Modality**:
  - Cloud Pattern Classification: Multi-class probabilities across 5 sky categories.
  - Rainfall Nowcasting: Discrete precipitation bin (`no_measurable_rain`, `light`, `moderate`, `heavy`) and continuous next-hour accumulated rainfall (mm).

---

## 2. Intended Use & Target Users
- **Intended Use**: Machine vision academic demonstration, educational sky observation, and localized short-term precipitation risk estimation.
- **Out of Scope / Non-Goals**:
  - Official national weather agency forecasts.
  - Safety-critical flood warning or disaster management.
  - Global numerical weather prediction from scratch.

---

## 3. Training & Evaluation Data
- **Vision Classes**:
  1. `clear_sky` (Clear blue sky gradient)
  2. `patterned_clouds` (Rippled altocumulus/stratocumulus)
  3. `thick_dark_clouds` (Dense dark cumulonimbus/nimbostratus)
  4. `thick_white_clouds` (Bright reflective cumulus)
  5. `veil_clouds` (Cirrostratus translucent sheet)
- **Splitting Strategy**:
  - Vision Model: Stratified split (70% train, 15% validation, 15% test).
  - Rainfall Model: Date-aware group split (samples from the same day remain together in a single split to avoid near-duplicate temporal leakage).
- **Label Source**: Explicitly logged as `open_meteo_proxy_reanalysis` (reanalysis proxy targets for prototype evaluation).

---

## 4. Quantitative Performance Summary (Held-Out Test Set)

### Task A: Cloud Pattern Classification
- **Held-Out Test Accuracy**: **77.78%**
- **Balanced Accuracy**: **80.00%**
- **Macro-F1 Score**: **0.7200**
- **Weighted-F1 Score**: **0.7111**
- **Per-Class F1 Scores**:
  - `clear_sky`: 1.0000 (Precision 100%, Recall 100%)
  - `patterned_clouds`: 1.0000 (Precision 100%, Recall 100%)
  - `thick_white_clouds`: 1.0000 (Precision 100%, Recall 100%)
  - `veil_clouds`: 0.6000 (Precision 42.9%, Recall 100%)
  - `thick_dark_clouds`: 0.0000 (Confused primarily with dense veil overcast)

### Task B: Next-Hour Rainfall Nowcasting
| Model Variant | Macro-F1 | Accuracy | MAE (mm) | RMSE (mm) |
|---|---|---|---|---|
| **Majority/Median Baseline** | 0.1667 | 50.0% | 3.88 mm | 6.94 mm |
| **Image-Only CNN Embeddings** | **0.2559** | 40.0% | **2.83 mm** | **4.53 mm** |
| **Multimodal (Vision + Weather)** | **0.3145** | 40.0% | N/A | N/A |

**Key Finding**:
The image feature embedding model significantly reduced rainfall regression Mean Absolute Error from **3.88 mm** (baseline) down to **2.83 mm**, and adding prediction-time weather features increased classification Macro-F1 from **0.2559** to **0.3145**.

---

## 5. Scientific Limitations & Ethical Considerations
1. Ground sky imagery observes bottom optical features. True precipitation physics depend heavily on vertical cloud thickness and atmospheric thermodynamic stability.
2. Models evaluated on localized proxy data cannot be generalized globally without on-site calibrated tipping-bucket rain gauge validation.
3. The UI explicitly separates Task A from Task B and marks proxy predictions with a dedicated provenance badge.
