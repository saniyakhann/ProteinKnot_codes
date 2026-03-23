# ProteinKnot_codes

Code repository for the MPhys thesis *"Do Local Writhe Matrices Encode Gauss Integral 
Descriptors for CATH Protein Classification?"* (University of Edinburgh, 2026).

This project investigates whether signed topological representations of protein backbone 
geometry (local writhe matrices) outperform unsigned pairwise Cα distance matrices for 
hierarchical protein fold classification across the four levels of the CATH hierarchy: 
Class (C), Architecture (A), Topology (T), and Homologous superfamily (H).

---

## Repository structure
```
ProteinKnot_codes/
├── Data_prep/
│   ├── data_filtering/
│   │   ├── single_domain_proteins_filtering.py
│   │   └── filter_missing_residues.py
│   ├── data_analysis/
│   │   ├── explore_cath_heirarchy.py
│   │   └── writhe_distribution_analysis.py
│   ├── CATHdata_no_lw.py
│   ├── distance_matrices.py
│   ├── downsampling.py
│   ├── gauss_updated.py
│   └── padding.py
├── MLmodel/
│   ├── Writhe_Matrix_Classifier/
│   │   ├── C_writhe.py
│   │   ├── A_writhe.py
│   │   ├── T_writhe.py
│   │   ├── H_writhe.py
│   │   └── inference_writhe.py
│   └── Distance_Matrix_Classifier/
│       ├── C.py
│       ├── A.py
│       ├── T.py
│       ├── H.py
│       └── Inference_distance.py
├── Interpretability/
│   ├── grad_cam.py
│   ├── dar_writhe_C.py
│   ├── dar_boxplot_writhe_only_C.py
│   ├── dar_writhe_A.py
│   ├── dar_boxplot_writhe_only_A.py
│   ├── dar_writhe_T.py
│   ├── dar_boxplot_writhe_only_T.py
│   ├── corrupted_context_ablation_T.py
│   ├── dar_writhe_H.py
│   └── dar_boxplot_writhe_only_H.py
├── Troubleshooting/
│   ├── analysis_of_writhematrix.py
│   └── check_pbd.py
├── Protein_visualisation.py
├── feature_vectors.py
└── pca_analysis_CATHlabel.py
```

---

## Data preparation (`Data_prep/`)

### data_filtering/
- `single_domain_proteins_filtering.py` — filters the CATH dataset to retain 
single-domain proteins suitable for matrix computation
- `filter_missing_residues.py` — removes proteins with missing Cα residues that would 
produce incomplete matrices

### data_analysis/
- `explore_cath_heirarchy.py` — exploratory analysis of the CATH label distribution 
across all four hierarchical levels
- `writhe_distribution_analysis.py` — analyses the distribution of writhe values 
across the dataset

### Root-level data preparation scripts
- `gauss_updated.py` — computes local writhe matrices using the discrete Gauss integral 
formulation; implements the pairwise segment crossing calculation using Numba JIT 
compilation for performance; outputs raw writhe matrices as compressed .npz files
- `CATHdata_no_lw.py` — downloads PDB structures from the RCSB, extracts Cα 
coordinates for a random sample of 1,000 single-domain proteins, computes writhe 
matrices using `gauss_updated.py`, and saves them to `Writhe_Matrices_Clean/`
- `padding.py` — pads all writhe matrices in `Writhe_Matrices_Clean/` to a uniform 
size determined by the largest matrix in the dataset, saving results to 
`Writhe_Matrices_Padded/`
- `downsampling.py` — downsamples all padded writhe matrices from 
`Writhe_Matrices_Padded/` to a fixed 256×256 resolution using block mean pooling, 
saving results to `Writhe_Matrices_256/`
- `distance_matrices.py` — downloads PDB structures from the RCSB, extracts Cα 
coordinates, computes pairwise Euclidean distance matrices, downsamples to 256×256 
via block mean pooling, and saves to `Distance_Matrices_256/`

---

## ML models (`MLmodel/`)

Both classifier families use the same CNN backbone. The Class-level models are standard 
CNNs taking only the matrix as input. The Architecture, Topology, and Homologous 
superfamily models are hierarchical CNNs that additionally receive normalised 
ground-truth labels from all preceding CATH levels as context inputs via a small MLP 
branch.

### Writhe_Matrix_Classifier/
- `C_writhe.py` — trains the Class-level writhe CNN on 256×256 writhe matrices
- `A_writhe.py` — trains the Architecture-level hierarchical writhe CNN with 
normalised Class label as context
- `T_writhe.py` — trains the Topology-level hierarchical writhe CNN with normalised 
Class and Architecture labels as context
- `H_writhe.py` — trains the Homologous superfamily-level hierarchical writhe CNN 
with normalised Class, Architecture, and Topology labels as context
- `inference_writhe.py` — runs full four-level cascade inference on a test set using 
predicted rather than ground-truth labels as context at each stage

### Distance_Matrix_Classifier/
- `C.py` — trains the Class-level distance CNN on 256×256 distance matrices
- `A.py` — trains the Architecture-level hierarchical distance CNN
- `T.py` — trains the Topology-level hierarchical distance CNN
- `H.py` — trains the Homologous superfamily-level hierarchical distance CNN
- `Inference_distance.py` — runs full four-level cascade inference for the distance 
model family

---

## Interpretability (`Interpretability/`)

Grad-CAM and DAR analysis is applied to the writhe model family at all four CATH 
levels. At the Class level only, paired writhe and distance saliency maps are generated 
for selected individual proteins.

- `grad_cam.py` — generates side-by-side Grad-CAM attention maps for a 
single specified protein comparing writhe and distance representations; set `domain_id` 
and `true_class` at the top of the script before running
- `dar_writhe_C.py` — computes the Diagonal Attention Ratio (DAR) for all Class-level 
test proteins using the writhe model; outputs `gradcam_results_writhe_C.csv`
- `dar_boxplot_writhe_only_C.py` — generates the DAR box plot grouped by CATH Class 
for the Class-level writhe model; outputs `DAR_boxplot_writhe_only_C.png`
- `dar_writhe_A.py` — computes DAR for Architecture-level test proteins using the 
writhe model with Class context; outputs `gradcam_results_writhe_A.csv`
- `dar_boxplot_writhe_only_A.py` — generates the DAR box plot grouped by Architecture 
class; outputs `DAR_boxplot_writhe_only_A.png`
- `dar_writhe_T.py` — computes DAR for Topology-level test proteins using the writhe 
model with Class and Architecture context; outputs `gradcam_results_writhe_T.csv`
- `dar_boxplot_writhe_only_T.py` — generates the DAR box plot grouped by CATH Class 
at the Topology level; outputs `DAR_boxplot_writhe_only_T.png`
- `corrupted_context_ablation_T.py` — replaces ground-truth Class and Architecture 
context inputs with randomly sampled incorrect values and evaluates both writhe and 
distance models over ten independent corruptions per protein; outputs 
`corrupted_context_ablation_T.csv`
- `dar_writhe_H.py` — computes DAR for Homologous superfamily-level test proteins 
using the writhe model with Class, Architecture, and Topology context; outputs 
`gradcam_results_writhe_H.csv`
- `dar_boxplot_writhe_only_H.py` — generates the DAR box plot grouped by CATH Class 
at the H level; outputs `DAR_boxplot_writhe_only_H.png`

---

## Troubleshooting (`Troubleshooting/`)

- `analysis_of_writhematrix.py` — checks writhe matrices for NaN values and 
asymmetry, reporting the location and extent of any problems found
- `check_pbd.py` — tests connectivity to the RCSB PDB by attempting to download a 
small set of test structures

---

## Other scripts
This are not directly relevant to the project and were initial codes I wrote to understand and experiment with the data

- `Protein_visualisation.py` — interactive script that accepts a CATH domain ID, 
downloads the corresponding PDB structure, and produces a three-panel figure showing 
the 3D backbone, the Cα pairwise distance matrix, and the XY projection
- `feature_vectors.py` — extracts a 15-dimensional feature vector from each writhe 
matrix, including statistical moments, diagonal and off-diagonal means, entropy, and 
normalised complexity measures
- `pca_analysis_CATHlabel.py` — performs PCA on the writhe feature vectors and 
produces scatter plots coloured by all four CATH hierarchy levels, with component 
loading analysis

---

## Dependencies
```
python >= 3.10
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
biopython
numba
opencv-python
requests
tqdm
```

---

## Notes on data

Writhe matrices and distance matrices are not included in this repository due to size. 
PDB structures are downloaded directly from the RCSB PDB during matrix computation. 
The protein list used for training and evaluation is derived from the CATH database 
v4.3.
