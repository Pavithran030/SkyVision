# Data Directory Structure

- `raw/`: Place raw downloaded sky image datasets here (e.g. SWIMCAT, TJNU GCD, or custom sky image folders). This folder is ignored by Git to avoid committing restricted or large dataset files.
- `processed/`: Intermediate processed assets, resized caches, or features.
- `manifests/`: CSV metadata manifests containing image paths, cloud classes, split assignments, timestamps, and paired rainfall labels (if available).

## Datasets Supported
1. **SWIMCAT**: Singapore Whole sky IMaging CATegories (5 classes: clear sky, patterned clouds, thick dark clouds, thick white clouds, veil clouds).
2. **TJNU GCD**: Ground-based Cloud Dataset (7 classes).
3. **Custom Sky Images**: Directory organized by class folders or paired timestamp manifests.
