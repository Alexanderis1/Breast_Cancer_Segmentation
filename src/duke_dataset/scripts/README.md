# Duke Breast Cancer MRI — 3D Tumour Detection

Multimodal deep learning model for automatic 3D bounding box localisation of breast cancer from DCE-MRI volumes, clinical data, and radiomic features.

**Dataset:** [Duke Breast Cancer MRI — TCIA](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)  
**Task:** Predict `[x₁, y₁, z₁, x₂, y₂, z₂]` bounding box in normalised [0,1] space  
**Metric:** 3D Intersection over Union (IoU) · Target: ≥ 0.55  
**Current best:** val IoU ~0.25 (v7 in progress with ResNet34 backbone)

---

## Results

| Version | Val IoU | Key change |
|---------|---------|------------|
| v2 (baseline) | 0.031 | Original code |
| v3 | 0.115 | Fix 1 (aug sync) + Fix 2 (DIoU head) |
| v4 | 0.165 | Fix 3 (positional encoding) |
| v5 | ~0.197 | DEPTH=64, IMAGE_SIZE=192, regularisation |
| v6 | ~0.250 | Laterality feature, 200 epochs |
| v7 | in progress | ResNet34 backbone |

---

## Architecture

```
Input: volume [B, 1, 64, 192, 192] + clinical [B, 15] + imaging [B, 13]

SliceEncoder (ResNet34, per-slice)   →  [B, 64, 512]
+ Positional embedding               →  [B, 64, 512]   
SliceAttention (weighted pool)       →  [B, 512]
CrossAttention (tabular → volume)    →  [B, 128]
Fusion MLP                           →  [B, 128]
BBox head (sorted sigmoid)           →  [B, 6]          
```

**Loss:** SmoothL1 + IoU + GIoU + DIoU + center distance  
**Backbone:** ResNet34 pretrained on ImageNet (grayscale-adapted: RGB conv1 averaged to 1 channel)

---

### Augmentation label desync 
Volume flips were sampled inside `_build_volume` without updating the bbox label. ~50% of training samples had contradictory image↔label pairs. Fixed by moving flip decisions to `__getitem__` and mirroring bbox coordinates to match.

**Impact: val IoU 0.031 → 0.115 (+271%)**

### Mean-box collapse 
The center+size decode parameterisation converged to predicting a constant average box for all patients. Fixed by replacing with a direct 6-output sigmoid head and adding 3D DIoU loss (gradient exists even at zero overlap).

**Impact: predictions became diverse and patient-specific**

### No Z-axis awareness 
`SliceAttention` treated all depth slices as an unordered bag — Z-localisation was pure guessing. Fixed by adding learnable positional embeddings (`nn.Embedding(128, 512)`) shared between SliceAttention and CrossAttention.

**Impact: val Z-dist 0.115 → 0.072; val IoU 0.115 → 0.165**

---

## Data Pipeline

### Series selection
Each patient has ~629 DICOMs across 6 series. The pipeline selects SeriesNumber 601 (first post-contrast DCE phase — peak tumour enhancement), falling back to the series with the most slices. DICOMs are sorted by `InstanceNumber` (alphabetical sort was an early bug).

### Volume construction
142 slices at 512×512 → subsampled to 64 slices → resized to 192×192 per slice → percentile clipped (p1–p99) → z-score normalised.

### Augmentation (training only)
- Horizontal, vertical, depth flips — all label-synced (Fix 1)
- In-plane rotation ±10° — bbox updated to minimum enclosing AABB
- Intensity scale jitter ×[0.85, 1.15]
- Gaussian noise σ=0.02

### Features
**Clinical (15 features):** age, menopause, days to MRI, scanner params (field strength, TR, TE, slice thickness), TNM staging, tumour grade, ER/PR/HER2, breast laterality (from `Tumor Location` JSON field).

**Radiomic (13 features):** tumour axis length, volume, GLCM texture (energy, contrast, homogeneity), DCE kinetics (peak enhancement, time to peak, uptake/washout), breast density, peak SER, elongation.

---

## Setup

```bash
pip install torch torchvision opencv-python matplotlib scikit-learn pillow pydicom pynrrd
```

### Data structure
```
exported_patients/
  train/
    Patient_002/
      patient_data.json        ← clinical features + annotations
      MRI_DICOM_sample/        ← DICOM files
        *.dcm
  test/
    ...
```

### Training

Open `duke_bbox_train.ipynb` and run all cells in order. The notebook:
1. Validates ground truth coverage
2. Loads dataset with laterality feature
3. Performs weight surgery warm-start from v6 checkpoint
4. Trains for up to 200 epochs with early stopping (patience=30)
5. Evaluates and saves prediction visualisations

### Configuration (Cell 16)

```python
DEPTH       = 64     # depth slices per volume
IMAGE_SIZE  = 192    # XY resolution per slice
BATCH_SIZE  = 4      # reduce to 2 if OOM
NUM_EPOCHS  = 200
PATIENCE    = 30
V6_CKPT     = "best_3d_bbox_model_v6.pth"  # warm-start source
```
---

## Weight Surgery Convention

Each notebook version preserves as much trained state from previous training. 

---

## Next Steps

1. **Evaluate** — ResNet34 results. Expected val IoU 0.35–0.50.
2. **Multi-scale features** — concatenate layer3 (256-d) + layer4 (512-d) before attention for finer XY resolution.
3. **Tumour size auxiliary loss** — regress `TumorMajorAxisLength_mm` as a physical calibration constraint on box size.
4. **Classification head** — add molecular subtype and grade prediction as auxiliary tasks.

---
