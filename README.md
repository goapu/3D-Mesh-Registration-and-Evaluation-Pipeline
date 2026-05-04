
# 3D Mesh Registration and Evaluation Pipeline

**Author:** Dilip Goswami

This repository contains a 3D mesh registration and evaluation pipeline for aligning a preoperative anatomical mesh to an intraoperative surface mesh. The pipeline estimates the forward transformation from preoperative space to intraoperative space, then applies the inverse transformation to bring the intraoperative model back into the original preoperative millimeter space for evaluation.

---

## Scripts

### `landmark_scale_ransac_icp.py`

Performs preoperative-to-intraoperative registration using:

- anatomical landmark centroid extraction from OBJ annotation meshes
- landmark-spread-based scale normalization
- rigid Kabsch alignment
- mesh-level RANSAC refinement
- ICP refinement with convergence logging
- forward and inverse transformation export

The final inverse-mapped intraoperative mesh is saved as:

```text
registered_model.obj
```

This mesh is in the original preoperative millimeter coordinate space.

### `distanceEvaluation.py`

Evaluates registration quality after inverse mapping using:

* one-way vertex-to-surface distance
* optional symmetric surface distance
* optional vertex-index correspondence evaluation from Excel
* summary metrics: mean, median, RMSE, P90, P95, P99, and threshold percentages
* histogram and CDF plots
* MeshLab-compatible colored error heatmap

---

## Registration Workflow

The main registration direction is:

```text
preoperative mesh → intraoperative mesh space
```

The total forward transformation is:

```text
T_total = T_icp @ T_ransac @ T_affine @ T_scale
```

The inverse transformation is then applied to the original intraoperative `model.obj`:

```text
registered_model = inverse(T_total) @ model
```

This produces `registered_model.obj`, which is used for evaluation against the original preoperative mucosa mesh.

---

## Installation

Create and activate a Python environment:

```bash
conda create -n meshreg python=3.10
conda activate meshreg
```

Install dependencies:

```bash
pip install numpy scipy trimesh open3d matplotlib pandas
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
numpy
scipy
trimesh
open3d
matplotlib
pandas
```

---

## Input Folder Structure

Example preoperative folder:

```text
preop_folder/
├── mucosa.obj
├── choanal_arc.obj
├── anterior*.obj
├── posterior*.obj
├── middle*.obj
├── inferior_turbinate*.obj
└── middle_turbinate*.obj
```

Example intraoperative folder:

```text
intraop_folder/
├── model.obj
├── choanal_arc.obj
├── anterior*.obj
├── posterior*.obj
├── middle*.obj
├── inferior_turbinate*.obj
└── middle_turbinate*.obj
```

The script identifies files using anatomical keywords in the filenames.

---

## Run Registration

```bash
python landmark_scale_ransac_icp.py \
  --preop "/path/to/preop_folder" \
  --intraop "/path/to/intraop_folder" \
  --output "/path/to/registration_output"
```

Main registration outputs:

```text
registration_output/
├── registered_model.obj
├── mucosa_registered_affine.obj
├── mucosa_registered_ransac.obj
├── mucosa_registered_icp.obj
├── scale.npy
├── affine.npy
├── ransac.npy
├── icp.npy
├── inv_scale.npy
├── inv_affine.npy
├── inv_ransac.npy
├── inv_icp.npy
├── inv_total.npy
├── transforms.npz
├── icp_convergence.csv
└── icp_convergence.png
```

---

## Run Evaluation

```bash
python distanceEvaluation.py \
  --preop_obj "/path/to/preoperative_mucosa.obj" \
  --intra_obj "/path/to/registration_output/registered_model.obj" \
  --output_dir "/path/to/evaluation_output"
```

For symmetric evaluation:

```bash
python distanceEvaluation.py \
  --preop_obj "/path/to/preoperative_mucosa.obj" \
  --intra_obj "/path/to/registration_output/registered_model.obj" \
  --output_dir "/path/to/evaluation_output" \
  --surface_mode symmetric
```

Evaluation outputs:

```text
evaluation_output/
├── summary_metrics_surface.csv
├── histogram_surface.png
├── cdf_surface.png
├── registered_model_heatmap.obj
└── heatmap_legend.txt
```

Open `registered_model_heatmap.obj` in MeshLab and enable:

```text
Render → Show Vertex Colors
```

---

## Optional Correspondence Evaluation

If an Excel file with 1-based vertex-index correspondences is available, run:

```bash
python distanceEvaluation.py \
  --preop_obj "/path/to/preoperative_mucosa.obj" \
  --intra_obj "/path/to/registration_output/registered_model.obj" \
  --output_dir "/path/to/evaluation_output" \
  --mapping_xlsx "/path/to/correspondences.xlsx"
```

The first two Excel columns should contain:

```text
preoperative_vertex_index, intraoperative_vertex_index
```

Both indices are expected to be 1-based.

---

## Example Result

One-way surface-distance evaluation:

```text
N samples: 19,157
Mean: 1.2051 mm
Median: 0.9635 mm
RMSE: 1.5571 mm
P95: 3.1606 mm
Max: 6.8763 mm

52.0% within 1 mm
80.8% within 2 mm
94.1% within 3 mm
99.7% within 5 mm
```

---

## Notes

* Registration is estimated from preoperative space to intraoperative space.
* Evaluation is performed after inverse mapping in the original preoperative millimeter space.
* Input meshes should be in OBJ format.
* Evaluation assumes both meshes are already in the same coordinate system.
* Distances are reported in the same units as the mesh coordinates, assumed to be millimeters.
* One-way surface distance is recommended for partial intraoperative reconstructions.
* No clinical or patient data is included in this repository.

---

## Usage

This repository is shared for demonstration and portfolio purposes. Reuse, redistribution, or modification is not permitted without the author’s permission.

---

## Recommended `.gitignore`

```gitignore
__pycache__/
*.pyc
.DS_Store

# Local data and generated outputs
data/
outputs/
results/
*.obj
*.ply
*.stl
*.npy
*.npz
*.csv
*.xlsx
*.png

# Environments
.venv/
venv/
env/
```

---

## Author

```
Dilip Goswami
M.Sc. Geodesy and Geoinformation Science
Technische Universität Berlin
```


