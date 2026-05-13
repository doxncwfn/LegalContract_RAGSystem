# Natural Language Processing - Legal Contract Pipeline

This repository implements the assignment flow from `docs/Specification.pdf`:

- **Assignment 1:** clause splitting, noun phrase chunking (IOB), dependency analysis
- **Assignment 2:** custom legal NER, semantic role labeling (SRL), clause intent classification

The source code lives in `src/`.

## 1) Environment Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

Notes:

- `en_core_web_sm` is used for Assignment 1 parsing tasks.
- `en_core_web_md` is used in `src/train_ner.py` for NER fine-tuning.
- SRL uses HuggingFace model `deepset/minilm-uncased-squad2` (downloaded automatically on first run).

## 2) Project Inputs / Outputs

- Input contracts: `src/input/*.txt`
- Main outputs: `src/output/`

Key output artifacts:

- `*_clauses.txt`
- `*_chunks.txt`
- `*_dependency.json`
- `ner_results.json`
- `srl_results.json`
- `intent_classification_final.json`

## 3) Re-run Full Pipeline (Recommended)

Use the unified runner:

```bash
python src/run_full_pipeline.py --input SPONSORSHIP_AGREEMENT.txt
```

Behavior:

- If `src/ner_model/` is missing, the pipeline automatically runs `src/train_ner.py` first.
- SRL is included by default.

Optional (skip SRL to run faster and avoid transformer download):

```bash
python src/run_full_pipeline.py --input SPONSORSHIP_AGREEMENT.txt --skip-srl
Force retrain NER even if a model already exists:
```

```bash
python src/run_full_pipeline.py --input SPONSORSHIP_AGREEMENT.txt --force-train-ner
```

You can replace the input with:

- `DISTRIBUTOR_AGREEMENT.txt`
- `SOFTWARE_LICENSE_AGREEMENT.txt`

## 4) Train Custom NER Model (if needed)

If you want to retrain NER from `src/ner_train_spacy.json`:

```bash
cd src
python train_ner.py
```

This will regenerate:

- `src/train.spacy`
- `src/ner_model/`

Then inference can be run via:

```bash
cd src
python predict_ner.py
python srl.py
```

## 5) File Overview

- `src/run_full_pipeline.py`: end-to-end pipeline for Assignment 1 + 2
- `src/train_ner.py`: spaCy fine-tuning script for legal NER
- `src/predict_ner.py`: NER inference over clause file
- `src/srl.py`: QA-based SRL over NER output

## 6) Reproducibility Notes

- Scripts use UTF-8 input/output.
- Paths are relative to `src/` data layout expected by the assignment.
- Existing generated artifacts in `src/output/` can be overwritten by reruns.
