# The Semantic Gap in Behavioral Embeddings
### Why Linear Methods Fail for Educational RAG in Mathematics

**EDM 2026** | Ricky Gole, Jamell Dacon | Morgan State University

---

## Overview

This repository contains the complete reproducibility pipeline for the paper. We audit the assumption that behavioral failure patterns in student data reveal semantic structures useful for tutoring. Using 15.8 million student interactions from UK secondary mathematics, we show that a 50-dimensional SVD behavioral manifold is functionally independent of neural semantic embeddings (Pearson r ≈ 0.000, 95% CI [−0.002, 0.001]).

We then introduce Manifold-Augmented Generation (MAG), a hybrid architecture enforcing semantic filtering before behavioral personalization, achieving 4.78/5 diagnostic quality versus 3.36/5 for behavioral-only retrieval.

---

## Repository Structure

```
semantic-gap-educational-rag/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── README.md              # Instructions to download required datasets
├── src/
│   ├── manifold.py            # Stage 1: SVD behavioral manifold construction
│   ├── clustering.py          # Stages 2-4: spectral clustering and labeling
│   ├── rag.py                 # Stage 6: RAG database with FAISS
│   ├── evaluate.py            # Stage 8: four-system comparative evaluation
│   ├── statistics.py          # Stages 9-10: paper statistics and correlation
│   └── figures.py             # Figure 2 and Figure 4 generation
├── scripts/
│   └── run_pipeline.py        # Main entry point
├── results/
│   └── hybrid_evaluated_complete.json   # Original evaluation results
└── notebooks/
    └── full_pipeline.ipynb    # Self-contained Colab notebook version
```

---

## Installation

```bash
git clone https://github.com/Rickygole/semantic-gap-educational-rag.git
cd semantic-gap-educational-rag
pip install -r requirements.txt
```

**Runtime environment:** Python 3.10+, Google Colab with T4 GPU recommended for Stage 2 spectral clustering.

---

## Data

Download the two required datasets and place them in the `data/` directory. See `data/README.md` for exact instructions.

* NeurIPS 2020 Education Challenge: __[https://eedi.com/projects/neurips-education-challenge](https://www.eedischool.com/projects/neurips-education-challenge)__ (direct download: __https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip__)
* Eedi Misconceptions: __https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics__
* NeurIPS 2020 Education Challenge: 15.8M student interactions
* Eedi Mining Misconceptions: 1,869 expert-labeled questions
---

## Usage

### Option A - Run the full pipeline from the command line

```bash
python scripts/run_pipeline.py \
    --data_dir data/ \
    --output_dir results/ \
    --openai_key YOUR_OPENAI_KEY
```

Skip stages that have already been run:

```bash
python scripts/run_pipeline.py \
    --data_dir data/ \
    --output_dir results/ \
    --openai_key YOUR_OPENAI_KEY \
    --skip_manifold \
    --skip_clustering \
    --skip_eval
```

### Option B - Run the Colab notebook

Upload `notebooks/full_pipeline.ipynb` to Google Colab, set runtime to T4 GPU, and run all cells in order.

### Option C - Reproduce paper statistics only

The original evaluation results are included at `results/hybrid_evaluated_complete.json`. To reproduce all Table 1 statistics without re-running the expensive API evaluation:

```python
import json
from src.statistics import compute_all_stats, compute_semantic_gap, print_paper_summary

with open("results/hybrid_evaluated_complete.json") as f:
    results = json.load(f)

stats = compute_all_stats(results)
print_paper_summary(stats)
```

---

## Key Results

| System | Quality (1-5) | 95% CI | Win Rate |
|--------|--------------|--------|----------|
| System D: MAG (Hybrid) | 4.78 | [4.69, 4.86] | 49.2% |
| System A: Zero-shot (GPT-4o-mini) | 4.27 | [4.06, 4.47] | 32.5% |
| System B: Semantic RAG | 3.94 | [3.74, 4.15] | 24.0% |
| System C: Behavioral RAG | 3.36 | [3.13, 3.60] | 12.3% |

**Semantic Gap:** Pearson r ≈ 0.000, 95% CI [−0.002, 0.001], p = 0.861

Win rate = pairwise tournament score. Random baseline for a 4-system tournament = 33.3%.

**Evaluation note:** Systems A, B, C were scored using GPT-4o-mini as judge. System D was scored using GPT-4o as judge. See paper Section 5.5 for discussion of this asymmetry.

---

## Estimated Runtime and Cost

| Stage | Description | Time | API Cost |
|-------|-------------|------|----------|
| 1 | SVD manifold | 10-15 min | - |
| 2-4 | Clustering | 5-10 min | - |
| 5-6 | Split + RAG | 5 min | - |
| 7 | DPO pairs | 20-30 min | ~$0.20 |
| 8 | Evaluation | 2-3 hours | ~$3-5 |
| 9-11 | Statistics | 5 min | - |

---

## Citation

```bibtex
@inproceedings{gole2026semanticgap,
  title     = {The Semantic Gap in Behavioral Embeddings: Why Linear Methods Fail for Educational RAG in Mathematics},
  author    = {Gole, Ricky and Dacon, Jamell},
  booktitle = {Proceedings of the 19th International Conference on Educational Data Mining},
  year      = {2026}
}
```

---

## License

MIT License. See `LICENSE` for details.
