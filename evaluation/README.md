# Evaluation

This folder contains the evaluation pipeline for the PII masking system.

## Files

- sample_annotations.txt – Sanitised sample of the custom annotated dataset.
- parse_annotations.py – Converts inline annotations into JSON.
- evaluate.py – Runs evaluation on the custom dataset.
- evaluate_ai4privacy.py – Runs detection evaluation on [ai4privacy](chatgpt://generic-entity?number=0).
- prepare_gum_docs.py and conll_parser.py – Utilities for PERSON grouping evaluation on [Georgetown University](chatgpt://generic-entity?number=1).

The full custom dataset is not included because it contains private conversational data.
