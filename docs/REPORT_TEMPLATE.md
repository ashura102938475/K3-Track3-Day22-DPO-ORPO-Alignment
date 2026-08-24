# Preference Alignment Experiment Report

- **Student**: NGUYỄN ANH TRÀ
- **Student ID**: 2A202601735

## 1. Dataset Analysis and Cleaning

### Data Loading Summary

- **Total examples loaded**: 24 labeled preference pairs.
- **Original data defect**: Line 1 contained unescaped double quotes around `self-attention`, so
  the original JSONL failed to parse at column 36. Escaping those quotes repaired the record.
- **Validation and normalization**: The loader reports the file and physical line number for JSON
  and schema errors and rejects normalized duplicate prompts. Required text is stripped and must
  remain non-empty. Duplicate/equality checks normalize text with Unicode NFKC, `casefold`, and
  whitespace splitting/joining, so Unicode-equivalent, case-only, and whitespace-only variants
  cannot bypass validation. Unknown top-level example fields are rejected.

### Split Strategy

- **Train/validation ratio**: The default 80/20 utility split gives 19 train and 5 validation
  prompt groups for this sample. No test partition is defined.
- **Leakage and reproducibility**: The utility groups by normalized prompt, sorts the normalized
  keys, then shuffles them with a local seeded RNG (default seed 42). Sorting before shuffling
  removes the earlier dependence on nondeterministic set iteration. Whole prompt groups stay in a
  single partition, every input row is retained exactly once, and row order within each partition
  is preserved.

## 2. DPO and ORPO Implementation

### Objective Selection

DPO is the primary reference-anchored objective: it favors the policy's chosen-vs-rejected
log-ratio over the frozen reference model's corresponding ratio. ORPO is also implemented as an
SFT mean plus a weighted odds-ratio preference penalty.

- `beta`: 0.1 by default; finite and strictly greater than zero.
- `lambda_orpo`: 0.1 by default; finite and greater than or equal to zero.

### Input Contracts

- **DPO** converts real numeric inputs to `float64` and rejects complex values before casting. All
  four log-probability arrays must be non-empty,
  finite, exactly the same shape, and at most zero. Exact-shape validation intentionally rejects
  arrays that NumPy could silently broadcast. The scaled margin must also remain finite.
- **ORPO** requires non-empty, finite chosen and rejected log-probability arrays with exactly the
  same shape and values strictly below zero. Zero is excluded because `log(1 - exp(logp))` and
  therefore log-odds are singular at `logp = 0`. SFT NLL values must be non-empty, finite, and
  non-negative; their mean is computed independently from the preference-batch mean. The
  preference weight may be zero.

### Numerical Stability

Both objectives evaluate log-sigmoid as `-np.logaddexp(0, -x)`. ORPO does not clip valid
log-probabilities. Large non-negative loss terms are scaled before their mean is reduced, avoiding
overflow when multiple finite terms approach the `float64` maximum. ORPO evaluates
`log(1 - exp(logp))` piecewise: for values above `-log(2)` it uses
`log(-expm1(logp))`, which is accurate near zero; otherwise it uses `log1p(-exp(logp))`, which is
stable in the negative tail. A regression with chosen/rejected log-probabilities `-100` and `-50`
has a penalty of approximately `50`; swapping them to the preferred ordering gives approximately
`1.9287498479639178e-22`. A value at the
closest representable negative float to zero also remains finite. Closed-form fixtures produce a
DPO loss of `0.663597` and an ORPO loss of `1.017086`.

### Configuration and Reproducible Environment

Configuration is parsed into nested typed Pydantic models. Unknown fields are forbidden at every
configuration level, training values are constrained, the method is limited to `dpo`, `orpo`, or
`mock`, and relative data/output/regression paths resolve from the configuration file's directory.
The native workflow is `uv sync --locked --extra dev` followed by `uv run ...`; CI uses the same
lockfile-based path. Python 3.11 is pinned in `.python-version` and is the tested environment.
Project metadata declares support for Python `>=3.10`, but this report does not claim that every
eligible Python version was tested. The only edit to the optional `trainers.py` outside a student
TODO is blank-line formatting required by Ruff; it has no behavioral effect.

## 3. Evaluation Results

### Exact Metrics Artifact

```json
{
  "evaluation_scope": "full_sample_lexical_baseline",
  "losses": 7,
  "mean_score_margin": 0.015940656565656557,
  "num_examples": 24,
  "pairwise_accuracy": 0.5208333333333334,
  "scorer": "lexical_overlap_v1",
  "tie_rate": 0.375,
  "ties": 9,
  "wins": 8
}
```

`lexical_overlap_v1` measures the fraction of unique prompt word tokens present in each response.
For a labeled pair, a chosen-response win receives one point and a tie receives half a point, so
the reported accuracy is `(8 + 0.5 * 9) / 24 = 0.5208333333333334`. These metrics cover all 24
labeled pairs in `data/sample_preferences.jsonl`. They are not held-out evaluation, trained-model
evaluation, before/after training comparison, or regression-prompt evaluation. No training run was
performed; the closed-form losses above are unit-test fixtures, not final training losses.

### Qualitative Failure Example

- **Prompt**: What is the purpose of a confusion matrix?
- **Chosen response**: A confusion matrix provides a detailed breakdown of classification
  performance, showing true positives, true negatives, false positives, and false negatives.
- **Rejected response**: A confusion matrix is used to visualize the distribution of the target
  variable in the dataset.
- **Scorer preference**: Incorrect. The rejected response scored 0.75 versus 0.50 because it
  repeats more prompt words even though its claim is factually wrong.

## 4. Discussion and Failure Modes

- **What worked**: Loading and configuration errors are contextual and concise; schema and metric
  boundaries are validated; splitting is deterministic and leak-free; and the DPO/ORPO losses
  match closed-form and extreme-value tests without relying on clipping.
- **Observed and fixed defects**: The original malformed JSON was repaired. Audit also found the
  baseline row-based split did not group equivalent prompts, whitespace-only text could pass
  validation, NumPy could silently broadcast mismatched loss arrays, and both DPO and ORPO were
  unimplemented. Normalized prompt grouping with sorted seeded keys, strip-before-length validation
  plus canonical normalization, exact-shape checks, and stable log-domain objectives address those
  defects.
- **Remaining correctness limitation**: Lexical overlap measures word reuse rather than factual
  quality. It loses on seven labeled pairs and ties on nine. The confusion-matrix example shows
  that a factually wrong response can win by echoing the prompt; response length can also create
  more opportunities for overlap. The score is useful only as a deterministic smoke-test baseline.
- **Safety gap**: The optional `PreferenceTrainer.train()` remains unimplemented and no generative
  model was run. Consequently, the four regression prompts cannot be marked pass or fail. A real
  evaluation must separately test high-risk medical redirection, strict summary limits,
  uncertainty calibration, and requests for missing troubleshooting context.
