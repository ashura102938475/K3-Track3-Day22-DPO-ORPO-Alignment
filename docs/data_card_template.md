# Data Card

- **Dataset name**: Preference Alignment Lab sample preferences
  (`data/sample_preferences.jsonl`).
- **Source and provenance**: 24 repository-provided English preference pairs about introductory
  machine-learning concepts. The original collection method, response generator, annotators, and
  adjudication process are not documented.
- **License and permission**: No dataset-specific license is declared in the repository. Treat the
  data as lab-only until reuse and redistribution permission is confirmed.
- **Unit and schema**: Each JSONL line is one labeled pair with required, non-empty string fields
  `prompt`, `chosen`, and `rejected`, plus an optional `metadata` object. The sample metadata uses
  `domain` and `rubric`; metadata contents are otherwise unconstrained. Unknown top-level fields
  are rejected rather than silently ignored.
- **Validation and normalization**: Required strings are stripped before the non-empty check.
  Prompt identity and chosen/rejected equality use Unicode NFKC normalization, `casefold`, and
  collapsed whitespace. Normalized duplicate prompts and normalized-equal responses are rejected;
  JSON and schema errors include file and physical line context. The original first record's
  unescaped quotes around `self-attention` were repaired.
- **Labeling rubric**: `chosen` is intended to be the more factually accurate educational answer;
  `rejected` is a plausible but incorrect or misleading alternative. Labels express a relative
  preference and are not absolute quality or safety scores.
- **Intended use**: Small, local exercises in preference-data validation, deterministic splitting,
  DPO/ORPO loss verification, and lexical-baseline evaluation. It is not intended for production
  model training, benchmark claims, or safety certification.
- **Train/validation/test split method**: Group records by normalized prompt, sort the normalized
  group keys, and shuffle them with a local seeded RNG (default seed 42) before assigning whole
  groups to the default 80/20 train/validation split. This sample yields 19 train groups and 5
  validation groups with no cross-partition prompt leakage. Original row order is preserved within
  each partition. No test partition exists.
- **Known biases and limitations**: The dataset is very small, English-only, synthetic-looking,
  limited to introductory ML education, and has no documented annotator disagreement or demographic
  coverage. It is not representative of open-domain preferences. Wording may leak labels, and the
  repository's full-sample lexical score measures prompt-token reuse rather than correctness; it is
  neither held-out nor model-based evaluation.
- **Safety and PII checks**: Manual inspection found no apparent personal data in the 24 records,
  but no automated PII detection or redaction is implemented. Automated checks cover structure,
  non-empty normalized text, response inequality, and duplicate prompts only; they do not establish
  privacy, factual accuracy, toxicity, or safety.
