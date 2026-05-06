# Educational Gender Bias Analysis Pipeline

This repository/documentation describes the full pipeline for analyzing item-level gender bias in educational materials, from raw assessment items to structured extraction outputs and final item-level bias scores.

The pipeline currently has two main components:

1. **GPT Extraction Pipeline**
   - reads educational assessment items
   - extracts person mentions
   - resolves coreference
   - infers entity-level gender
   - extracts mention-level linguistic features
   - writes scorer-ready mention tables

2. **Item-level Bias Scorer**
   - learns feature-value weights from a background corpus
   - scores extracted mentions in target items
   - aggregates mention-level contributions into item-level bias scores
   - writes item-level, mention-level, and feature-weight outputs

This README is intended to document the **entire end-to-end workflow**.

---

## End-to-End Workflow

The full workflow is:

1. prepare a CSV of educational items
2. run the GPT extraction pipeline
3. inspect the extracted outputs and validation logs
4. use the generated `mentions.csv` as scorer input
5. fit the scorer on a background mention corpus
6. score the target items
7. analyze:
   - item-level directional scores
   - item-level intensity scores
   - mention-level contributions
   - feature-value weights

---

## Pipeline Structure

A typical project layout looks like this:

```text
project/
├── input_items.csv
├── gpt_extraction_pipeline.py
├── scorer.py
├── README.md
├── extraction_outputs/
│   ├── extractions.jsonl
│   ├── mentions.csv
│   ├── entities.csv
│   └── errors.csv
└── scoring_outputs/
    ├── item_scores.csv
    ├── mention_scores.csv
    ├── feature_weights.csv
    └── scoring_config.json
```

---

# Part I. GPT Extraction Pipeline

## Purpose

The GPT extraction pipeline converts raw educational text into structured person-centered annotations suitable for downstream gender-bias scoring.

It is designed for short educational items such as:

- K–12 science assessment items
- textbook questions
- short instructional passages
- quiz-style items

## Input

The extractor expects a CSV file with at least:

- `item_id`
- `item_text`

Example:

```csv
item_id,item_text
item_001,"Maria placed one plant in sunlight and another in the shade. She measured their heights for two weeks."
item_002,"The mother boiled water in a pot while her son watched the steam rise."
```

## Extraction Tasks

For each item, the extractor performs:

1. **person mention extraction**
2. **coreference resolution**
3. **entity-level gender inference**
4. **mention-level feature extraction**

## Person Mentions

Person mentions include:

- proper names
- pronouns
- kinship terms
- person-denoting common nouns
- occupations and roles
- group references to people

Examples:

- `Maria`
- `She`
- `the mother`
- `his daughter`
- `the students`
- `a scientist`

## Coreference

Mentions are clustered into entities/coreference chains.

Example:

- `Maria` and `She` may belong to the same entity
- `the father` and `him` may belong to the same entity

## Gender Inference

Gender is assigned at the entity level using only textual evidence such as:

- pronouns
- kinship terms
- gendered nouns
- titles
- strongly gender-associated names when appropriate

Allowed labels:

- `M`
- `F`
- `N`
- `andy`
- `unknown`

The pipeline is designed **not** to infer gender from stereotypes, occupations alone, or world knowledge that is not grounded in the text.

## Mention-level Features

The extractor currently supports feature fields such as:

- `mention_type`
- `namedness`
- `kinship`
- `grammatical_role`
- `dep_pos`
- `root`
- `root_pos`
- `agency`
- `authority`
- `generic_he`
- `conj`
- `verb_negation`
- `number`
- `personhood_confidence`
- `coref_confidence`

These are designed to align with the scorer.

## Output Files

The extraction pipeline writes:

### 1. `extractions.jsonl`

Full structured output per item.

Includes:

- `item_id`
- `item_text`
- `mentions`
- `entities`
- `notes`
- `validation_errors`
- `autofix_messages` (if enabled)

### 2. `mentions.csv`

Flattened mention-level output.

This file is intended to be directly usable by the scorer.

Important columns:

- `item_id`
- `mention_id`
- `entity_id`
- `gender`
- `mention_reliability`
- `mention_text`
- `start`
- `end`
- `mention_type`
- `namedness`
- `kinship`
- `grammatical_role`
- `dep_pos`
- `root`
- `root_pos`
- `agency`
- `authority`
- `generic_he`
- `conj`
- `verb_negation`
- `number`

### 3. `entities.csv`

Flattened entity/coreference output.

Important columns:

- `item_id`
- `entity_id`
- `canonical_mention`
- `mention_ids`
- `gender`
- `gender_confidence`
- `gender_evidence_chain`

### 4. `errors.csv`

Exceptions and validation errors.

## Span Validation and Auto-fix

Because LLM offsets can be slightly wrong, the extractor includes validation and optional auto-fix logic.

Typical fixes include:

- trimming leading/trailing whitespace
- locally recovering the exact mention span near the predicted offset
- case-insensitive local recovery

This helps repair cases like:

- offsets yield `" me"` but mention text is `"She"`
- offsets yield `"he "` but mention text is `"she"`

## Notebook-safe Execution

The extractor can be run both from terminal and from notebooks/Colab.

In notebooks, use `run_batch(...)` directly instead of relying on CLI parsing.

## Console Logging

The extraction pipeline prints progress messages while processing, including:

- input file loading
- item ID and progress count
- short text preview
- when each LLM call starts
- when each LLM call finishes
- number of extracted mentions/entities
- validation status
- autofix messages
- output file locations

---

# Part II. Item-level Bias Scorer

## Purpose

The scorer converts extracted mention-level features into interpretable item-level bias scores.

The scorer is designed to work with:

- GPT extraction outputs
- rule-based extraction outputs
- manually prepared mention tables

The only strict requirement is that the target mention table contains:

- one row per mention
- a mention-level `gender` column
- the feature columns you want to use

## Inputs

The scorer uses two datasets:

### 1. Background corpus mention table

This file is used to learn gender-associated feature-value weights.

Each row should contain:

- `gender`
- one or more feature columns

Typical columns:

- `mention_id`
- `gender`
- `dep_pos`
- `root`
- `root_pos`
- `agency`
- `authority`
- `kinship`
- `namedness`
- `generic_he`
- `mention_reliability`

### 2. Target item mention table

This is usually the extractor’s `mentions.csv`.

Each row should contain:

- `item_id`
- `mention_id`
- `gender`
- the same feature columns used in training
- optionally `mention_reliability`

## Feature-Agnostic Design

The scorer is not hard-coded to a fixed set of features.

Instead, it takes a list of feature specifications. This allows the scorer to work with:

- syntactic features
- semantic features
- lexical features
- binary indicators
- future custom features

Supported feature types:

- `categorical`
- `binary`

## Core Scoring Idea

The scorer learns a weight for each feature-value pair using a background corpus.

For each feature-value pair `(f=v)`, it estimates:

\[
w_{f=v} = \log \frac{P(f=v \mid M)+\alpha}{P(f=v \mid F)+\alpha}
\]

where:

- `M` = male mentions
- `F` = female mentions
- `alpha` = smoothing constant

Interpretation:

- positive weight = more male-associated in the background corpus
- negative weight = more female-associated
- zero = no association

## Support and Shrinkage

To reduce instability from rare feature values, the scorer computes support and shrinks low-support weights toward zero.

Support is:

- the number of gender-resolved background mentions that exhibit a given feature value

Shrunk weight:

\[
\tilde{w}_{f=v} = \frac{n_{f=v}}{n_{f=v}+k} w_{f=v}
\]

where:

- `n_{f=v}` = support
- `k` = shrinkage parameter

## Mention-level Contributions

Each mention receives signed contributions from its active feature values.

The sign depends on:

- mention gender
- learned feature-value weight
- mention reliability
- feature reliability

## Item-level Scores

The scorer produces:

### 1. Raw score

Total signed evidence accumulated across mentions in an item.

Interpretation:

- negative = more stereotype-consistent
- positive = more counter-stereotypical

### 2. Normalized score

Raw score divided by total available weighted evidence.

Interpretation:

- near `-1` = most evidence points stereotypical
- near `+1` = most evidence points counter-stereotypical
- near `0` = weak or balanced evidence

### 3. Final score

Optional `tanh` transform of the normalized score.

### 4. Intensity score

Magnitude-based measure of how strongly gender-associated the item is, regardless of direction.

This distinguishes:
- weakly gender-coded items
from
- strongly gender-coded but balanced items

## Scorer Output Files

The scorer writes:

### 1. `item_scores.csv`

One row per item.

Typical columns:

- `item_id`
- `num_mentions`
- `raw_score`
- `normalized_score`
- `final_score`
- `intensity_score`
- `feature_contributions_json`

### 2. `mention_scores.csv`

One row per mention.

Typical columns:

- `item_id`
- `mention_id`
- `gender`
- `mention_reliability`
- `mention_score_raw`
- `mention_abs_score`
- `mention_evidence_total`
- `contributions_json`

### 3. `feature_weights.csv`

Learned feature-value weights from the background corpus.

Typical columns:

- `feature`
- `value`
- `n_m`
- `n_f`
- `support`
- `p_m`
- `p_f`
- `log_weight`
- `shrunk_weight`
- `reliability`

### 4. `scoring_config.json`

Saved scorer configuration and feature specification metadata.

---

# Part III. How the Two Components Connect

The extraction pipeline and scorer are designed to work together.

## Integration Requirement

The scorer expects mention-level rows with a `gender` column.

Earlier, the extraction output only stored gender at the entity level. This caused scorer failures like:

- `KeyError: 'gender'`

This was fixed by propagating entity-level gender to every mention row in `mentions.csv`.

## Scorer-ready Extraction Output

The extractor now writes scorer-ready mention rows with:

- `gender`
- `mention_reliability`
- all extracted feature columns needed by the scorer

So the standard workflow is:

1. run extraction
2. take `mentions.csv`
3. feed it directly into the scorer

---

# Part IV. Example End-to-End Usage

## Step 1. Prepare item input

```csv
item_id,item_text
item_001,"Maria placed one plant in sunlight and another in the shade. She measured their heights for two weeks."
item_002,"The mother boiled water in a pot while her son watched the steam rise."
```

## Step 2. Run extraction

Notebook/Colab example:

```python
from pathlib import Path

run_batch(
    input_csv=Path("/content/k12_science_assessment_items_100.csv"),
    output_dir=Path("/content/extraction_outputs"),
    model="gpt-5.4",
    item_id_col="item_id",
    text_col="item_text",
    max_items=5,
)
```

## Step 3. Fit the scorer

```python
import pandas as pd

background_df = pd.read_csv("background_mentions.csv")
target_df = pd.read_csv("extraction_outputs/mentions.csv")

feature_specs = [
    FeatureSpec(name="dep_pos", feature_type="categorical", reliability=1.0, min_support=5),
    FeatureSpec(name="root", feature_type="categorical", reliability=0.6, min_support=10),
    FeatureSpec(name="root_pos", feature_type="categorical", reliability=0.9, min_support=5),
    FeatureSpec(name="agency", feature_type="categorical", reliability=1.0, min_support=5),
    FeatureSpec(name="authority", feature_type="categorical", reliability=1.0, min_support=5),
    FeatureSpec(name="conj", feature_type="categorical", reliability=0.8, min_support=5),
    FeatureSpec(name="verb_negation", feature_type="binary", reliability=0.9, min_support=5),
    FeatureSpec(name="kinship", feature_type="binary", reliability=1.0, min_support=5),
    FeatureSpec(name="namedness", feature_type="binary", reliability=1.0, min_support=5),
    FeatureSpec(name="generic_he", feature_type="binary", reliability=1.0, min_support=5),
]

cfg = ScoringConfig(
    gender_col="gender",
    item_id_col="item_id",
    mention_id_col="mention_id",
    mention_reliability_col="mention_reliability",
)

scorer = BiasScorer(feature_specs=feature_specs, config=cfg)
scorer.fit(background_df)
item_scores_df, mention_scores_df = scorer.score_items(target_df)
```

## Step 4. Save scorer outputs

```python
save_scoring_outputs(
    scorer=scorer,
    item_scores_df=item_scores_df,
    mention_scores_df=mention_scores_df,
    output_dir="scoring_outputs",
    save_weights=True,
    save_config=True,
)
```

---

# Part V. Evaluation Considerations

The pipeline outputs should not be treated as automatically valid evidence of bias without evaluation.

A strong evaluation plan should include:

## 1. Extraction quality evaluation

Evaluate:

- person mention extraction
- coreference clustering
- gender inference
- feature extraction accuracy

## 2. Controlled minimal pairs

Construct nearly identical item pairs where only gender-relevant structure changes.

Examples:

- kinship flips
- subject/object role reversals
- pronoun swaps
- namedness manipulations

The score should change in the expected direction.

## 3. Human judgment evaluation

Sample items across the score range and ask human annotators to judge:

- whether stereotype cues are present
- direction
- strength
- supporting text

## 4. Robustness analysis

Test:

- background corpus variation
- feature ablations
- bootstrap resampling
- score stability under pipeline uncertainty

---

# Installation

## Extraction pipeline dependencies

```bash
pip install -U openai pydantic pandas
```

Set API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

In Colab:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

## Scorer dependencies

```bash
pip install pandas numpy
```

---

# Common Issues

## `KeyError: 'gender'`

Cause:
- mention-level scorer input does not contain a mention-level `gender` column

Fix:
- use the patched extraction pipeline that propagates entity gender to each mention row

## Offset mismatch errors

Examples:

- `offsets yield ' me' but mention text is 'She'`
- `offsets yield 'he ' but mention text is 'she'`

Fix:
- use validation and span auto-fix in the extraction pipeline

## Notebook `argparse` crash

Cause:
- CLI argument parsing in notebook/Colab

Fix:
- use `run_batch(...)` directly instead of the CLI main block

## Sparse lexical features

Cause:
- rare feature values produce unstable weights

Fix:
- keep shrinkage enabled
- use minimum support thresholds
- lower lexical feature reliability if necessary

---

# Recommended Workflow Summary

1. prepare educational items in CSV format
2. run GPT extraction
3. inspect `errors.csv`
4. use `mentions.csv` as scorer input
5. fit scorer on background mentions
6. score items
7. save scorer outputs
8. inspect item and mention contributions
9. evaluate score validity with human and controlled tests

---

# Files Included in This Pipeline

Typical files you may have:

- `gpt_extraction_pipeline.py`
- `scorer.py`
- `README.md`
- `k12_science_assessment_items_100.csv`
- `background_mentions.csv`
- extraction outputs
- scoring outputs

---

# Summary

This pipeline provides an end-to-end framework for analyzing item-level gender bias in educational text.

It is designed to:

- extract person mentions from raw educational items
- resolve coreference and infer gender
- derive scorer-ready mention-level features
- learn feature-value gender associations from a background corpus
- compute interpretable item-level bias scores
- save all intermediate and final outputs for inspection and reproducibility

The pipeline is modular, extensible, and suitable for research-oriented experimentation on educational bias.
