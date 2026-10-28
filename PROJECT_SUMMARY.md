# DFU Diagnostic Benchmark Project Summary

## 1) Active model architectures

The application currently loads five independent prediction members for diabetic foot ulcer screening:

| Model | Architecture | Final parameter count | Benchmark accuracy |
|---|---|---:|---:|
| MobileNetV2 | `mobilenet_v2` | 2,226,434 | 97.51% |
| EfficientNet-B0 | `efficientnet_b0` | 4,010,110 | 97.75% |
| DenseNet-121 | `densenet121` | 6,955,906 | 97.16% |
| Swin Transformer | `swin_t` | 27,520,892 | 98.70% |
| Random Forest | `RandomForestClassifier` | 100 trees (ensemble, not a dense parameter tensor) | 93.36% |

Notes:
- The four CNN models are the primary decision makers because they are weighted more heavily in the ensemble than the Random Forest member.
- The app uses `MODEL_WEIGHTS` in `app.py` to weight each model during aggregation:
  - MobileNetV2: 0.24
  - EfficientNet-B0: 0.26
  - DenseNet-121: 0.24
  - Swin Transformer: 0.21
  - Random Forest: 0.05
- The benchmark accuracies are the values stored in the main application configuration for each loaded model.

## 2) Preprocessing and feature engineering

### 2.1 Skin/foot isolation

The app preprocesses every image before inference in `app.py`:

- Converts the input image to RGB.
- Uses a YCrCb + HSV skin mask to isolate the foot from the background.
- Applies morphological closing/opening and dilation to smooth the skin mask and reduce noise.
- Finds the largest contour corresponding to the skin region and crops around it to focus on the foot.
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space to normalize lighting and improve feature visibility.

This process is implemented by:
- `_skin_mask(...)`
- `isolate_foot(...)`

### 2.2 Patch extraction and multi-view analysis

The app does not classify the full foot in a single shot. Instead, it scans multiple tissue patches from the isolated foot:
- Full-image patch
- Top-left, top-right, bottom-left, bottom-right patches
- Center crop
- Three overlapping vertical strips (left, mid, right)

This is implemented by `analysis_views(...)` in `app.py` and allows small ulcers or plantar lesions to be captured even when they are not centered in the image.

### 2.3 Tissue risk cues

The app computes risk heuristic scores to help catch early/inflammatory or hidden DFU patterns when the wound is not obvious:

- `Erythema`: local and global redness score based on the LAB color channels, intended to detect inflammation and local redness.
- `Slough`: yellowish tissue cue derived from the blue channel and surrounding-tone differences; helps flag soft, devitalized tissue.
- `Dark tissue`: low-light / darkened tissue score based on local luminance values to approximate necrosis or devitalized tissue.
- `Texture`: edge/texture variation estimated from a Laplacian of the grayscale image to capture roughness or uneven surfaces.

These features are computed in `tissue_cues(...)`, which returns:
- `erythema`
- `slough`
- `necrosis`
- `texture`
- `risk`

The final risk score is a weighted combination of erythema, slough, and necrosis:

`risk = 0.55 * erythema + 0.30 * slough + 0.15 * necrosis`

### 2.4 Grad-CAM support

The Streamlit UI supports Grad-CAM heatmaps for explainability. `GradCAM` registers forward and backward hooks on a selected CNN layer and generates a class activation map indicating which image regions contributed most to the DFU prediction.

## 3) Pipeline changes and structural updates

### 3.1 Unified training pipeline

The repository now includes a consolidated training entry point, `train_unified.py`, which:
- detects the `data/dataset` directory automatically
- builds a `ImageFolder` dataset
- performs a random train/validation split
- trains each model individually via a single CLI entry
- saves model outputs under `models/` using unified file names such as:
  - `mobilenet_v2_unified.pth`
  - `efficientnet_b0_unified.pth`
  - `densenet121_unified.pth`
  - `swin_t_unified.pth`
  - `rf_unified.pkl`

`train_unified.py` includes:
- `get_data_loaders(...)`
- `build_and_freeze_model(...)`
- `train_pytorch_model(...)`
- `train_random_forest(...)`

### 3.2 Model loading and class mapping

The Streamlit app loads all trained models via `load_all_models()` and resolves the semantic class order using the saved checkpoint metadata:
- `class_to_idx`
- `healthy` / `Healthy` / `normal` mapping
- `ulcer` / `DFU` / `dfu` mapping

This allows the app to preserve the correct class semantics even when the saved checkpoint has a custom mapping.

### 3.3 Streamlit UI improvements

Recent app changes include:
- a single dashboard UI for upload/webcam/live-mode classification
- model cards with per-model probabilities and benchmark accuracy
- a side panel for threshold presets:
  - `Balanced (recommended)`
  - `High precision (fewer false DFU)`
  - `High recall (catch hidden ulcers)`
- Grad-CAM heatmap toggle
- skin isolation toggle
- tissue cue metrics displayed for erythema, slough, dark tissue, and texture
- final ensemble verdict with weighted diagnosis and confidence explanation

### 3.4 Decision logic refinements

The app’s verdict logic was tightened to reduce false positives from healthy feet:
- redness alone is not sufficient for a DFU diagnosis
- CNN consensus is required before calling a case DFU in ambiguous scenarios
- the ensemble verdict uses confidence from multiple models and only upgrades to DFU when sufficient evidence exists

## 4) Ensemble and inference strategy

### 4.1 Per-model probability generation

For each model:

- CNNs: `predict_pytorch(...)` takes all patch views and applies test-time augmentation by using the original crop and a left-right flipped version.
- The model outputs a softmax distribution over class logits, and the DFU probability is read from the `ulcer_idx` class.
- Random Forest: `predict_rf(...)` interpolates the patch to 64x64 and uses `predict_proba(...)` to obtain the DFU-class probability.

### 4.2 Patch aggregation within a model

Each model produces one DFU score per crop. The app then combines those patch-level scores into a single model score using `aggregate_patch_scores(...)`:

- `hot = number of patch scores >= hot_patch_threshold`
- `blended = 0.55 * mean(scores) + 0.45 * percentile_90(scores)`

This gives a single patched probability that balances average risk with the most suspicious hotspot.

### 4.3 Weighted ensemble aggregation

The final ensemble is a weighted sum of the per-model outputs:

- Each model has a fixed weight in `MODEL_WEIGHTS`
- Individual model DFU scores are combined as:
  - `weighted_sum = Σ(model_score * model_weight)`
  - `weight_total = Σ(model_weight)`
  - `ensemble_raw = weighted_sum / weight_total`

The app then adjusts the final DFU probability slightly based on tissue cues if the image shows necrotic or slough-like tissue, but it does not allow weak redness alone to override a healthy consensus.

### 4.4 Final verdict logic

`decide_verdict(...)` combines:
- CNN decision count (`n_pos`)
- strong DFU count (`n_strong`)
- mean CNN score (`mean_cnn`)
- tissue heuristic outputs (`lesion_cues`)

The app returns one of three final verdicts:
- `Healthy`
- `Possible inflammation`
- `DFU Detected`

The final output is designed to prioritize model consensus and avoid false positives from a single aggressive model or from non-ulcer visual cues such as redness or dark skin shading.

## Final project note

This project is structured as a research prototype for diabetic foot ulcer screening. It combines:
- deep learning model benchmarking
- image preprocessing and lesion cue estimation
- multi-view patch scanning
- weighted ensemble diagnosis
- Grad-CAM explainability

The current codebase is setup for iterative refinement: as more clinically verified healthy and DFU examples are added to `data/dataset`, the model ensemble can be retrained and re-evaluated in the same workflow.
