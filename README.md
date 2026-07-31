# AG News Topic Classification: TF-IDF Baseline vs Fine-Tuned DistilBERT

Comparing a classical sparse-feature baseline against a fine-tuned BERT-style transformer on four-way news topic classification, with statistical significance testing, error analysis, and a cost/efficiency comparison.

> Research Methods (7PAM2015) — Assignment 3, MSc Data Science

## Overview

- **Task:** 4-way topic classification — World / Sports / Business / Sci-Tech
- **Dataset:** [AG News](https://huggingface.co/datasets/ag_news) (Hugging Face Hub), ~127,600 short news articles from 2,000+ sources
- **Baseline:** TF-IDF (uni+bi-grams) + `LinearSVC`
- **Model:** `distilbert-base-uncased`, fine-tuned via Hugging Face `Trainer`

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| TF-IDF + LinearSVC (baseline) | 0.9014 | 0.9009 | 0.9014 | 0.9010 |
| **Fine-tuned DistilBERT** | **0.9220** | **0.9221** | **0.9220** | **0.9220** |

- Fine-tuning improved macro-F1 by **2.10 points**, a gain confirmed **statistically significant** by McNemar's test (χ² = 48.243, p < 0.000001).
- DistilBERT (67.0M parameters) took **328.9s** to fine-tune vs **7.0s** for the baseline (~47x), and ran inference at **605 rows/sec** vs **4,625 rows/sec** for the baseline (~7.6x slower).
- Both models struggle most on **Business ↔ Sci/Tech** confusion, reflecting genuine topical overlap (tech-company finance/product news) rather than a fixable modelling gap.

Full write-up with methodology, per-class breakdown, and limitations: see [`report/AG_News_Report.docx`](report/).

## Repository structure

```
.
├── notebooks/
│   └── agnews_topic_classification.ipynb   # full pipeline: EDA → baseline → fine-tuning → evaluation
├── report/
│   └── AG_News_Report.docx                 # written report (methodology, results, discussion)
├── outputs/
│   ├── figures/                            # generated plots (confusion matrices, learning curves, etc.)
│   ├── tables/                             # generated CSVs (comparison tables, per-class metrics)
│   └── models/final/                       # saved fine-tuned model + tokenizer
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

**requirements.txt**
```
transformers>=4.46,<5
datasets>=2.19
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
accelerate>=0.30
tabulate>=0.9
statsmodels>=0.14
torch
pandas
numpy
```

## Usage

Run the full pipeline end to end (recommended: Google Colab with a GPU runtime):

```bash
jupyter notebook notebooks/agnews_topic_classification.ipynb
```

Or open directly in Colab: **File → Upload notebook** → select `agnews_topic_classification.ipynb` → set Runtime to GPU → **Run all**.

Total runtime: ~15–20 minutes on a single T4 GPU (training data is stratified-subsampled to 20k rows by default; set `SETTINGS["train_subsample"] = None` in the notebook to use the full 120k-row training set).

## Methodology summary

- **Preprocessing:** stratified train/validation/test split, deduplication, leakage checks, WordPiece tokenization with `max_length=64` (justified from the 99th percentile of measured token lengths).
- **Baseline:** TF-IDF (unigrams+bigrams) + `LinearSVC`, regularisation strength selected via grid search on validation macro-F1.
- **Fine-tuning:** AdamW optimizer, linear LR schedule with warmup, cross-entropy loss, 3 epochs, learning rate 2e-5, dynamic batch padding.
- **Evaluation:** accuracy, macro precision/recall/F1, per-class metrics, ROC-AUC, confusion matrices, McNemar's significance test, and a training/inference cost comparison.

## Limitations

- Trained on a 20,000-row stratified subsample of the full 120,000-row training set to fit a constrained time budget.
- Baseline ROC-AUC uses uncalibrated `LinearSVC` decision-function scores, not true probabilities.
- Single run with a fixed seed (23); metrics are not averaged across multiple runs.
- AG News is short, English-only, and predates recent news events/terminology.

## References

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108.
- Zhang, X., Zhao, J., LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015.
- McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika, 12(2).

## License

This project is for academic coursework purposes (7PAM2015, Research Methods in Data Science).
