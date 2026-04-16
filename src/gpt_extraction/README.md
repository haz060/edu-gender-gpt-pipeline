# GPT Extraction Pipeline for Person Mentions, Coreference, Gender, and Linguistic Features

This pipeline uses GPT to process educational assessment items and produce structured annotations that can be used directly for downstream gender-bias scoring.

The pipeline is designed for short science-style assessment items, textbook passages, and similar educational materials. For each item, it extracts person mentions, resolves coreference, infers gender at the entity level, derives mention-level linguistic features, and writes flattened outputs that can be passed directly into the scorer.

---

## Overview

Given an input CSV of assessment items, the pipeline:

1. reads each item of text
2. sends the text to GPT with a structured extraction prompt
3. extracts:
   - person mentions
   - coreference chains
   - entity-level gender labels
   - mention-level linguistic features
4. auto-fixes certain offset mismatches when possible
5. validates the structured output
6. writes:
   - a full JSONL file
   - a mention-level CSV
   - an entity-level CSV
   - an error log CSV

The mention-level CSV includes propagated entity gender, so it can be used directly by the scoring code.

---

## Input Format

The input file should be a CSV with at least these columns:

- `item_id`
- `item_text`

Example:

```csv
item_id,item_text
item_001,"Maria placed one plant in sunlight and another in the shade. She measured their heights for two weeks."
item_002,"The mother boiled water in a pot while her son watched the steam rise."
```

---

## Output Files

The pipeline writes four files to the output directory.

### 1. `extractions.jsonl`

One JSON record per item containing the full structured extraction output.

Each record includes:

- `item_id`
- `item_text`
- `mentions`
- `entities`
- `notes`
- `autofix_messages` (if enabled)
- `validation_errors`

### 2. `mentions.csv`

Flattened mention-level output. This file is intended to be directly usable by the scorer.

Important columns include:

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

Flattened entity-level output.

Important columns include:

- `item_id`
- `entity_id`
- `canonical_mention`
- `mention_ids`
- `gender`
- `gender_confidence`
- `gender_evidence_chain`

### 4. `errors.csv`

Contains exceptions and validation errors.

---

## Extraction Tasks

For each input item, the model is prompted to perform the following tasks:

### 1. Person mention extraction

The model identifies person-denoting mentions, including:

- proper names
- pronouns
- kinship terms
- common nouns referring to people
- occupations and role nouns
- group references to people

### 2. Coreference resolution

Mentions referring to the same person or people are grouped into an entity/coreference chain.

### 3. Gender inference

Gender is inferred at the entity level using only textual evidence such as:

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

### 4. Mention-level feature extraction

The pipeline extracts a set of linguistic and semantic features for each mention, including:

- mention type
- namedness
- kinship
- grammatical role
- dependency-like position
- associated root
- root POS
- agency
- authority
- generic masculine usage
- conjunction/connective
- predicate negation
- number
- confidence values

---

## Direct Compatibility with the Scorer

The scorer expects a mention-level CSV with a `gender` column and feature columns. To support this, the pipeline propagates entity-level gender to every mention row in `mentions.csv`.

This means the output can be used directly with scorer code configured like this:

```python
cfg = ScoringConfig(
    gender_col="gender",
    item_id_col="item_id",
    mention_id_col="mention_id",
    mention_reliability_col="mention_reliability",
)
```

---

## Offset Auto-fix

LLM-generated offsets may occasionally be slightly misaligned. For example:

- the span includes extra whitespace
- the start/end is shifted by a few characters
- the extracted substring does not exactly match the mention text

The pipeline can automatically repair many of these cases by:

1. trimming whitespace from the predicted span
2. searching locally around the predicted offset for the exact mention text
3. optionally using a case-insensitive local search

If a span is repaired successfully, the corrected offsets are used for downstream processing.

This is especially useful for errors like:

- offsets yield `" me"` but mention text is `"She"`
- offsets yield `"he "` but mention text is `"she"`

---

## Validation

After extraction, the pipeline validates:

- span bounds are legal
- extracted substring matches mention text
- every mention refers to a known entity
- every entity lists valid mention IDs
- mention IDs are unique
- entity IDs are unique
- mentions are sorted by start offset
- propagated gender is present and valid

Validation errors are written to:

- `validation_errors` in `extractions.jsonl`
- `errors.csv`

---

## Installation

Install the required packages:

```bash
pip install -U openai pydantic pandas
```

Set your API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

In Colab:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

---

## Running the Pipeline

### From a terminal

```bash
python gpt_extraction_pipeline.py \
  --input_csv /path/to/input.csv \
  --output_dir /path/to/output_dir \
  --model gpt-5.4 \
  --item_id_col item_id \
  --text_col item_text
```

### In a notebook or Colab

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

Or:

```python
main([
    "--input_csv", "/content/k12_science_assessment_items_100.csv",
    "--output_dir", "/content/extraction_outputs",
    "--model", "gpt-5.4",
    "--item_id_col", "item_id",
    "--text_col", "item_text",
    "--max_items", "5",
])
```

---

## Console Messages

The pipeline prints progress messages while processing, including:

- batch start
- number of loaded rows
- current item ID
- preview of the item text
- when the LLM request starts
- when the LLM request finishes
- number of mentions and entities returned
- validation status
- autofix status
- errors and exceptions
- output file locations

Example:

```text
[START] Loading input CSV from: /content/k12_science_assessment_items_100.csv
[INFO] Loaded 100 rows
================================================================================
[PROCESS] Item 1/5 | item_id=item_001
[TEXT] Maria placed one plant in sunlight and another in the shade. She measured their heights...
[LLM] Sending request for item_id=item_001 ...
[LLM] Completed item_id=item_001 in 4.82s | mentions=2 entities=1
[VALIDATION] item_id=item_001 passed
[DONE] item_id=item_001 finished in 4.83s
```

---

## Expected Mention-Level Columns

The mention-level output is designed to support bias scoring. Typical columns include:

- `item_id`
- `mention_id`
- `entity_id`
- `gender`
- `mention_reliability`
- `gender_confidence`
- `gender_evidence_chain`
- `item_text`
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
- `personhood_confidence`
- `coref_confidence`
- `gender_evidence_local`

---

## Model Assumptions

The extraction prompt is designed with the following assumptions:

- gender should be inferred only from textual evidence
- gender should not be inferred from stereotypes
- occupations alone should not determine gender
- entity-level gender is the authoritative label
- mention-level output should preserve exact surface forms when possible

---

## Recommended Workflow

A typical workflow is:

1. prepare assessment items in CSV format
2. run the GPT extraction pipeline
3. inspect `errors.csv` and validation messages
4. use `mentions.csv` as input to the scorer
5. analyze item-level bias scores downstream

---

## Common Issues

### `KeyError: 'gender'`

This happens when the scorer expects a mention-level `gender` column but the extraction output does not include it.

This pipeline fixes that by propagating entity-level gender to each mention row.

### Offset mismatch errors

Examples:

- `offsets yield ' me' but mention text is 'She'`
- `offsets yield 'he ' but mention text is 'she'`

These can often be repaired automatically by the autofix step.

### Notebook `argparse` crash

If the script is run inside Colab or Jupyter with the CLI entrypoint, `argparse` may fail because required command-line arguments were not provided.

Use `run_batch(...)` directly in notebooks.

---

## Possible Extensions

This pipeline can be extended with:

- retry logic for failed API calls
- checkpointing and resume support
- tqdm progress bars
- more aggressive offset recovery
- additional extracted features
- multiple-pass extraction
- human correction interface integration

---

## File Structure Example

```text
project/
├── gpt_extraction_pipeline.py
├── input_items.csv
└── extraction_outputs/
    ├── extractions.jsonl
    ├── mentions.csv
    ├── entities.csv
    └── errors.csv
```

---

## Summary

This GPT extraction pipeline converts educational assessment items into structured, mention-level and entity-level annotations for downstream gender-bias analysis.

It is designed to:

- extract person mentions
- resolve coreference
- infer gender carefully
- produce scorer-ready mention rows
- repair common offset issues
- provide transparent validation and logging
