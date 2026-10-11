# Experiment Log

| Date | Task | Model / Setup | Split / Data | Metric 1 | Metric 2 | Notes |
|------|------|---------------|--------------|----------|----------|-------|
| 2026-10-11 | Scaffold | Project Initialization | N/A | Tests passing (18/18) | N/A | Modular repository structure, tests and configs created |
| 2026-10-11 | Data Pipeline | Dataset Inspector & Manifest | 120 sky images (5 classes) | 120 valid, 0 corrupt | 0 duplicates | Stratified train (84), val (18), test (18) |
| 2026-10-11 | Task A: Cloud CNN | MobileNetV3-Small (Transfer Learning) | Train: 84, Val: 18 | Val Acc: 83.33% | Val Macro-F1: 0.7455 | AdamW (lr=1e-3, 5 epochs), best checkpoint saved |
| 2026-10-11 | Task A: Evaluation | MobileNetV3-Small on Held-Out Test Split | Test: 18 | Test Acc: 77.78% | Test Macro-F1: 0.7200 | Balanced Acc: 80.00%. Confusion matrix saved |
| 2026-10-11 | Task B: Pairing | Timestamp-to-Rainfall Alignment | 120 paired frames (12 dates) | Date-aware split: 80/20/20 | 0 date leakage | Target interval: [t, t+1h), source: open_meteo_proxy_reanalysis |
| 2026-10-11 | Task B: Baseline | Majority Classifier + Median Regressor | Test: 20 | Test Macro-F1: 0.1667 | Test MAE: 3.88 mm | Baseline benchmark |
| 2026-10-11 | Task B: Image ML | Random Forest on CNN Embeddings | Test: 20 | Test Macro-F1: 0.2559 | Test MAE: 2.83 mm | Outperforms baseline in MAE (2.83 vs 3.88 mm) and Macro-F1 (0.2559 vs 0.1667) |
| 2026-10-11 | Task B: Multimodal | Random Forest (Embeddings + Weather Context) | Test: 20 | Test Macro-F1: 0.3145 | Balanced Acc: 40.0% | Multimodal context improves classification Macro-F1 |
