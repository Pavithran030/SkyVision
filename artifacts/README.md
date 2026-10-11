# Model Artifacts Directory

Trained model checkpoints, feature extractors, label encoders, and metadata are saved here:
- `cloud_classifier_best.pt`: PyTorch checkpoint containing model state dict, architecture, class mapping, and normalization parameters.
- `rainfall_classifier_best.joblib`: Scikit-learn model for next-hour rainfall category prediction.
- `rainfall_regressor_best.joblib`: Scikit-learn model for next-hour continuous rainfall amount (mm).
- `multimodal_classifier_best.joblib`: Scikit-learn model combining CNN image embeddings with meteorological features.
- `model_metadata.json`: Artifact manifest detailing training date, git commit/version, validation scores, and label sources.
