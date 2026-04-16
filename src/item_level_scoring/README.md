# Item-level Bias Scorer

This README describes the item-level bias scoring code used to compute directional and intensity-based bias scores from extracted person mentions and their linguistic features.

The scorer is designed to work with mention-level rows, where each row corresponds to one extracted person mention in an educational item. It supports arbitrary extracted features, learns feature-value weights from a background corpus, and produces item-level scores that can be interpreted as directional stereotype scores and bias intensity scores.

---

## Overview

The scorer takes two main inputs:

1. a **background corpus mention table**
2. a **target item mention table**

Using the background corpus, it estimates feature-value associations with gender. It then applies those learned weights to the target item mentions and aggregates mention-level contributions into item-level scores.

The scorer is:

- feature-agnostic
- extensible to new extracted features
- compatible with mention-level extraction outputs
- designed to work with propagated mention-level gender labels
- interpretable at both mention and item levels

---

## Main Idea

The scoring pipeline works in five stages:

1. define the feature columns to use
2. estimate background weights for feature values by gender
3. optionally shrink unreliable low-support weights
4. compute mention-level contributions
5. aggregate mention contributions into item-level scores

The resulting item-level outputs include:

- a **raw score**
- a **normalized score**
- an optional **final score** after `tanh`
- an **intensity score**
- per-feature contribution summaries

---

## Core Concepts

### 1. Background weights

For each feature-value pair, the scorer estimates a log-association weight using the background corpus:

\[
w_{f=v} = \log \frac{P(f=v \mid M)+\alpha}{P(f=v \mid F)+\alpha}
\]

where:

- `M` = male mentions
- `F` = female mentions
- `alpha` = smoothing constant

Positive values indicate the feature value is more associated with male mentions in the background corpus. Negative values indicate stronger association with female mentions.

### 2. Support

Support means the number of gender-resolved mentions in the background corpus that exhibit a particular feature value.

For example:

- if `dep_pos=nsubj` appears 120 times for male mentions and 80 times for female mentions,
- then the support for `dep_pos=nsubj` is `200`.

Support is used for shrinkage so that rare feature values do not get overly strong weights.

### 3. Shrinkage

The scorer optionally shrinks low-support weights toward zero:

\[
\tilde{w}_{f=v} = \frac{n_{f=v}}{n_{f=v}+k} w_{f=v}
\]

where:

- `n_{f=v}` = support
- `k` = shrinkage hyperparameter

### 4. Mention contribution

For each mention and each active feature value, the scorer computes a signed contribution. The contribution depends on:

- the mention gender
- the learned feature-value weight
- feature reliability
- mention reliability

### 5. Item-level aggregation

Mention contributions are summed across the item to obtain an item-level score.

---

## Scores Produced by the Scorer

### Raw score

The raw score is the total signed evidence accumulated across all mentions and active feature values in an item.

Interpretation:

- negative = more stereotype-consistent evidence
- positive = more counter-stereotypical evidence

The raw score is useful but depends on item size and total amount of evidence.

### Normalized score

The normalized score divides the raw score by the total available weighted evidence in the item.

Interpretation:

- near `-1` = most evidence points in the stereotypical direction
- near `+1` = most evidence points in the counter-stereotypical direction
- near `0` = either weak evidence or balanced evidence

This is generally the most interpretable directional score across items.

### Final score

If enabled, the scorer applies:

\[
\tanh(\gamma \cdot S(i))
\]

to the normalized score, producing a bounded final score.

### Intensity score

The intensity score measures the magnitude of gender-associated evidence regardless of direction.

Interpretation:

- high intensity = strongly gender-coded item
- low intensity = weakly gender-coded item

This is helpful because an item may have a directional score near zero but still contain strong positive and negative evidence that cancel out.

---

## Input Requirements

The scorer expects mention-level tabular input.

### Background corpus file

The background file should contain one row per mention with:

- a `gender` column
- one or more feature columns

Example columns:

- `mention_id`
- `source_doc_id`
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

### Target item file

The target item file should also contain one row per mention with:

- `item_id`
- `mention_id`
- `gender`
- the same feature columns used in the background file

Optional:

- `mention_reliability`

The GPT extraction pipeline produces mention-level output in a format that can be used directly by the scorer after entity gender has been propagated to each mention row.

---

## Expected Target Mention Columns

Typical scorer-compatible mention columns include:

- `item_id`
- `mention_id`
- `gender`
- `mention_reliability`
- `dep_pos`
- `root`
- `root_pos`
- `agency`
- `authority`
- `conj`
- `verb_negation`
- `kinship`
- `namedness`
- `generic_he`

You can add more extracted feature columns later without changing the scoring logic, as long as you define them in the feature specification list.

---

## Feature-Agnostic Design

The scorer is designed so that you do not need to hard-code a fixed set of features such as:

- `kinship`
- `agency`
- `authority`
- `dep_pos`
- `root`

Instead, you define a list of feature specifications, and the scorer automatically learns weights for whatever features you provide.

This makes it easy to extend the model when new extracted features become available.

---

## Main Classes

### `FeatureSpec`

Defines one feature family.

Typical fields:

- `name`
- `feature_type`
- `active_values`
- `missing_values`
- `reliability`
- `min_support`
- `normalize_strings`

Feature types supported:

- `categorical`
- `binary`

### `ScoringConfig`

Defines global scorer settings such as:

- `gender_col`
- `item_id_col`
- `mention_id_col`
- `mention_reliability_col`
- `alpha`
- `shrink_k`
- `epsilon`
- whether to use `tanh`
- whether to normalize by evidence

### `BiasScorer`

Main scorer class.

Main methods:

- `fit(background_df)`
- `export_feature_weights()`
- `score_mentions(target_df)`
- `score_items(target_df)`

---

## Example Usage

### 1. Define feature specifications

```python
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
]
```

### 2. Initialize the scorer

```python
cfg = ScoringConfig(
    gender_col="gender",
    item_id_col="item_id",
    mention_id_col="mention_id",
    mention_reliability_col="mention_reliability",
    alpha=0.5,
    shrink_k=10.0,
    use_tanh=False,
    use_evidence_normalization=True,
)

scorer = BiasScorer(feature_specs=feature_specs, config=cfg)
```

### 3. Fit on background mentions

```python
import pandas as pd

background_df = pd.read_csv("background_mentions.csv")
scorer.fit(background_df)
```

### 4. Inspect learned weights

```python
weights_df = scorer.export_feature_weights()
print(weights_df.head(20))
```

### 5. Score target items

```python
target_df = pd.read_csv("mentions.csv")
item_scores_df, mention_scores_df = scorer.score_items(target_df)

print(item_scores_df.head())
print(mention_scores_df.head())
```

---

## Output Tables

### Feature weights table

The learned weights table typically contains:

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

### Mention scores table

The mention-level scoring output typically contains:

- `item_id`
- `mention_id`
- `gender`
- `mention_score_raw`
- `mention_abs_score`
- `mention_evidence_total`
- `contributions_json`

### Item scores table

The item-level scoring output typically contains:

- `item_id`
- `num_mentions`
- `raw_score`
- `normalized_score`
- `final_score`
- `intensity_score`
- `feature_contributions_json`

---

## Interpretation of Scores

### Directional interpretation

- negative = more stereotype-consistent
- positive = more counter-stereotypical

### Strength interpretation

- low intensity = weak gender signal
- high intensity = strong gender-associated signal

Useful combinations:

- low normalized, low intensity = weakly stereotypical, weak overall
- low normalized, high intensity = strongly stereotypical
- near-zero normalized, high intensity = mixed but strongly gendered
- near-zero normalized, low intensity = mostly neutral

---

## Why Use Both Direction and Intensity

A signed score alone can be misleading.

For example:

- one item may be near zero because it contains almost no gender-associated evidence
- another item may also be near zero because it contains strong positive and negative evidence that cancel out

The intensity score helps distinguish these two cases.

---

## Support and Reliability

The scorer uses support and reliability to make weights more stable.

### Support

Support is the number of observations of a feature value in the background corpus.

Higher support means:

- more empirical evidence
- more stable learned weights

Lower support means:

- noisier weights
- stronger shrinkage toward zero

### Mention reliability

A mention reliability value can be supplied per row, for example:

- `1.0` for explicit pronouns or high-confidence mentions
- lower values for uncertain mentions

The scorer can use this to discount noisy extracted mentions.

### Feature reliability

Each feature family can also be assigned a reliability weight.

This is useful because some feature families may be more stable or trustworthy than others.

---

## Compatible Files

The scorer can work with:

- hand-prepared mention tables
- rule-based extraction outputs
- GPT extraction outputs

The most important requirement is that the target mention file contains:

- one row per mention
- a usable `gender` column
- the relevant feature columns

---

## Example Background File

```csv
mention_id,source_doc_id,gender,dep_pos,root,root_pos,agency,authority,conj,verb_negation,kinship,namedness,generic_he,mention_reliability
bg_001,doc_a,M,nsubj,lead,VERB,high,high,none,0,0,1,0,1.0
bg_002,doc_a,F,dobj,help,VERB,low,low,none,0,1,0,0,1.0
bg_003,doc_b,M,nsubj,build,VERB,high,medium,and,0,0,1,0,0.9
bg_004,doc_b,F,nsubj,care,VERB,medium,low,and,0,1,0,0,1.0
```

---

## Example Target File

```csv
item_id,mention_id,gender,mention_reliability,dep_pos,root,root_pos,agency,authority,conj,verb_negation,kinship,namedness,generic_he
item_001,m1,F,1.0,nsubj,design,VERB,high,high,and,0,0,1,0
item_001,m2,M,1.0,pobj,assist,VERB,low,low,none,0,1,0,0
item_001,m3,F,1.0,nsubj,discover,VERB,high,high,none,0,0,0,0
```

---

## Installation

Install required packages:

```bash
pip install pandas numpy
```

If you are using the full scorer implementation discussed in the project, also ensure:

- Python 3.9+
- `dataclasses` support
- `typing` and `json` from the standard library

---

## Recommended Workflow

1. prepare or generate a background corpus mention file
2. run extraction on the target items
3. verify the target mention file includes `gender`
4. define the feature list
5. fit the scorer on the background file
6. score the target items
7. inspect item-level and mention-level outputs
8. analyze top contributing features

---

## Common Issues

### `KeyError: 'gender'`

This happens when the target mention file does not contain a mention-level `gender` column.

Fix:
- propagate entity-level gender to each mention row before scoring

### Sparse feature values

Some lexical feature values may appear only a few times in the background corpus.

Fix:
- keep shrinkage enabled
- set `min_support`
- reduce reliability for sparse lexical features

### Item length effects

Raw scores tend to increase with the number of mentions and active features.

Fix:
- use normalized scores for comparison across items

### Unknown or unresolved gender

If mentions have `gender = unknown`, the scorer may drop them depending on configuration.

Fix:
- check the scorer’s gender handling settings
- decide whether to exclude or retain uncertain mentions

---

## Possible Extensions

The scorer can be extended in several ways:

- additional extracted features
- different normalization schemes
- alternative shrinkage methods
- calibration of feature reliabilities
- separate models for different domains
- integration with item review tools
- visualization of top contributing features per item

---

## File Structure Example

```text
project/
├── scorer.py
├── background_mentions.csv
├── mentions.csv
└── outputs/
    ├── item_scores.csv
    ├── mention_scores.csv
    └── feature_weights.csv
```

---

## Summary

The item-level scorer converts mention-level extracted features into interpretable item-level bias scores.

It is designed to:

- learn feature-value weights from a background corpus
- score mentions and items using those weights
- handle arbitrary extracted features
- support reliability weighting and shrinkage
- produce both directional and intensity-based scores
- work directly with scorer-ready mention outputs from the extraction pipeline
