from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import math
import json

import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass
class FeatureSpec:
    """
    Specification for one feature family.

    name:
        Column name in the dataframe.

    feature_type:
        "categorical" or "binary".

    active_values:
        For categorical features: if provided, only these values are used.
        For binary features: values interpreted as "present" if in active_values.
        If None:
          - categorical -> all observed non-null values are used
          - binary -> default positive values are inferred from {1, True, "1", "true", "yes", "y"}

    missing_values:
        Values treated as missing and therefore ignored.

    reliability:
        Feature-family reliability weight lambda_f in [0,1].

    min_support:
        Optional minimum support threshold. Values below this support are ignored.

    normalize_strings:
        Whether to lowercase and strip strings before processing.
    """
    name: str
    feature_type: str = "categorical"
    active_values: Optional[List[Any]] = None
    missing_values: List[Any] = field(default_factory=lambda: [None, np.nan, "", "nan", "none", "unknown"])
    reliability: float = 1.0
    min_support: int = 1
    normalize_strings: bool = True


@dataclass
class ScoringConfig:
    """
    Global configuration for the scoring model.
    """
    gender_col: str = "gender"
    item_id_col: str = "item_id"
    mention_id_col: Optional[str] = "mention_id"
    mention_reliability_col: Optional[str] = None

    # Gender mapping
    male_values: Tuple[Any, ...] = ("M", "male", "Male", 1, "1")
    female_values: Tuple[Any, ...] = ("F", "female", "Female", -1, "-1")

    # Weight estimation
    alpha: float = 0.5     # smoothing
    shrink_k: float = 10.0 # support-based shrinkage
    epsilon: float = 1e-8

    # Optional final tanh transform
    use_tanh: bool = False
    tanh_gamma: float = 1.0

    # Whether to average raw mention contributions instead of using evidence normalization
    use_evidence_normalization: bool = True

    # If True, ignore mentions whose gender is not resolvable to M/F
    drop_unknown_gender: bool = True


# ============================================================
# Utilities
# ============================================================

def _is_missing(x: Any, missing_values: List[Any]) -> bool:
    if pd.isna(x):
        return True
    if x in missing_values:
        return True
    # string normalization for common missing markers
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null", "unknown"}:
        return True
    return False


def _normalize_value(x: Any, normalize_strings: bool = True) -> Any:
    if pd.isna(x):
        return None
    if isinstance(x, str) and normalize_strings:
        return x.strip().lower()
    return x


def _gender_sign(x: Any, cfg: ScoringConfig) -> Optional[int]:
    if x in cfg.male_values:
        return 1
    if x in cfg.female_values:
        return -1
    if isinstance(x, str):
        y = x.strip().lower()
        if y in {str(v).strip().lower() for v in cfg.male_values if isinstance(v, str)}:
            return 1
        if y in {str(v).strip().lower() for v in cfg.female_values if isinstance(v, str)}:
            return -1
    return None


def _binary_present(x: Any, active_values: Optional[List[Any]]) -> bool:
    if pd.isna(x):
        return False
    if active_values is not None:
        return x in active_values
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "present"}
    return bool(x)


# ============================================================
# Core scorer
# ============================================================

class BiasScorer:
    """
    Feature-agnostic item-level bias scorer.

    This class:
      1. learns feature-value weights from a background corpus,
      2. scores target mentions/items using those weights.

    The score is extensible to arbitrary features such as kinship,
    agency, authority, dependency role, namedness, etc.
    """

    def __init__(self, feature_specs: List[FeatureSpec], config: Optional[ScoringConfig] = None):
        self.feature_specs = {fs.name: fs for fs in feature_specs}
        self.config = config or ScoringConfig()

        # Learned statistics
        self.value_stats_: Dict[str, Dict[Any, Dict[str, float]]] = {}
        self.fitted_: bool = False

    # --------------------------------------------------------
    # Fitting background weights
    # --------------------------------------------------------

    def fit(self, background_df: pd.DataFrame) -> "BiasScorer":
        """
        Fit feature-value weights from a background corpus dataframe.

        background_df must contain:
          - gender column
          - feature columns named in feature_specs
        """
        df = background_df.copy()

        # Resolve gender
        df["_gsign"] = df[self.config.gender_col].apply(lambda x: _gender_sign(x, self.config))
        if self.config.drop_unknown_gender:
            df = df[df["_gsign"].isin([1, -1])].copy()

        self.value_stats_ = {}

        for feat_name, spec in self.feature_specs.items():
            if feat_name not in df.columns:
                raise ValueError(f"Feature column '{feat_name}' not found in background dataframe.")

            feat_stats: Dict[Any, Dict[str, float]] = {}

            if spec.feature_type == "categorical":
                series = df[feat_name].apply(lambda x: _normalize_value(x, spec.normalize_strings))
                valid_mask = ~series.apply(lambda x: _is_missing(x, spec.missing_values))
                tmp = df.loc[valid_mask, ["_gsign"]].copy()
                tmp["_value"] = series[valid_mask]

                if spec.active_values is not None:
                    allowed = {_normalize_value(v, spec.normalize_strings) for v in spec.active_values}
                    tmp = tmp[tmp["_value"].isin(allowed)]

                values = sorted(tmp["_value"].dropna().unique().tolist())

                for value in values:
                    n_m = int(((tmp["_value"] == value) & (tmp["_gsign"] == 1)).sum())
                    n_f = int(((tmp["_value"] == value) & (tmp["_gsign"] == -1)).sum())

                    total_m = int((tmp["_gsign"] == 1).sum())
                    total_f = int((tmp["_gsign"] == -1).sum())

                    p_m = (n_m + self.config.alpha) / (total_m + self.config.alpha * max(len(values), 1))
                    p_f = (n_f + self.config.alpha) / (total_f + self.config.alpha * max(len(values), 1))

                    log_weight = math.log(p_m / p_f)
                    support = n_m + n_f
                    shrink = support / (support + self.config.shrink_k)
                    shrunk_weight = shrink * log_weight

                    if support >= spec.min_support:
                        feat_stats[value] = {
                            "n_m": n_m,
                            "n_f": n_f,
                            "support": support,
                            "p_m": p_m,
                            "p_f": p_f,
                            "log_weight": log_weight,
                            "shrunk_weight": shrunk_weight,
                            "reliability": spec.reliability,
                        }

            elif spec.feature_type == "binary":
                series = df[feat_name]
                present_mask = series.apply(lambda x: _binary_present(x, spec.active_values))
                tmp = df.loc[:, ["_gsign"]].copy()
                tmp["_present"] = present_mask

                n_m = int((tmp["_present"] & (tmp["_gsign"] == 1)).sum())
                n_f = int((tmp["_present"] & (tmp["_gsign"] == -1)).sum())

                total_m = int((tmp["_gsign"] == 1).sum())
                total_f = int((tmp["_gsign"] == -1).sum())

                p_m = (n_m + self.config.alpha) / (total_m + 2 * self.config.alpha)
                p_f = (n_f + self.config.alpha) / (total_f + 2 * self.config.alpha)

                log_weight = math.log(p_m / p_f)
                support = n_m + n_f
                shrink = support / (support + self.config.shrink_k)
                shrunk_weight = shrink * log_weight

                if support >= spec.min_support:
                    feat_stats[True] = {
                        "n_m": n_m,
                        "n_f": n_f,
                        "support": support,
                        "p_m": p_m,
                        "p_f": p_f,
                        "log_weight": log_weight,
                        "shrunk_weight": shrunk_weight,
                        "reliability": spec.reliability,
                    }
            else:
                raise ValueError(f"Unsupported feature_type='{spec.feature_type}' for feature '{feat_name}'.")

            self.value_stats_[feat_name] = feat_stats

        self.fitted_ = True
        return self

    # --------------------------------------------------------
    # Inspect learned weights
    # --------------------------------------------------------

    def export_feature_weights(self) -> pd.DataFrame:
        """
        Export learned feature-value weights as a dataframe.
        """
        if not self.fitted_:
            raise RuntimeError("BiasScorer must be fitted before exporting weights.")

        rows = []
        for feat_name, feat_stats in self.value_stats_.items():
            for value, stats in feat_stats.items():
                rows.append({
                    "feature": feat_name,
                    "value": value,
                    **stats
                })
        if not rows:
            return pd.DataFrame(columns=[
                "feature", "value", "n_m", "n_f", "support", "p_m", "p_f",
                "log_weight", "shrunk_weight", "reliability"
            ])
        return pd.DataFrame(rows).sort_values(["feature", "support"], ascending=[True, False]).reset_index(drop=True)

    # --------------------------------------------------------
    # Mention scoring
    # --------------------------------------------------------

    def _get_mention_reliability(self, row: pd.Series) -> float:
        if self.config.mention_reliability_col and self.config.mention_reliability_col in row.index:
            val = row[self.config.mention_reliability_col]
            if pd.isna(val):
                return 1.0
            try:
                v = float(val)
                return max(0.0, min(1.0, v))
            except Exception:
                return 1.0
        return 1.0

    def score_mentions(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each mention in the target dataframe.

        Returns one row per mention with:
          - mention_score_raw
          - mention_abs_score
          - per-feature contribution details in JSON
        """
        if not self.fitted_:
            raise RuntimeError("BiasScorer must be fitted before scoring.")

        df = target_df.copy()
        df["_gsign"] = df[self.config.gender_col].apply(lambda x: _gender_sign(x, self.config))

        if self.config.drop_unknown_gender:
            df = df[df["_gsign"].isin([1, -1])].copy()

        rows = []

        for idx, row in df.iterrows():
            gsign = row["_gsign"]
            mention_reliability = self._get_mention_reliability(row)

            contribs = []
            mention_raw = 0.0
            mention_abs = 0.0
            evidence_total = 0.0

            for feat_name, spec in self.feature_specs.items():
                if feat_name not in row.index:
                    continue

                feat_stats = self.value_stats_.get(feat_name, {})

                if spec.feature_type == "categorical":
                    raw_val = row[feat_name]
                    value = _normalize_value(raw_val, spec.normalize_strings)
                    if _is_missing(value, spec.missing_values):
                        continue
                    if value not in feat_stats:
                        continue

                    stat = feat_stats[value]
                    weight = stat["shrunk_weight"]
                    lam = spec.reliability
                    contrib = -1.0 * gsign * mention_reliability * lam * weight

                    mention_raw += contrib
                    mention_abs += abs(contrib)
                    evidence_total += mention_reliability * lam * abs(weight)

                    contribs.append({
                        "feature": feat_name,
                        "value": value,
                        "gender_sign": gsign,
                        "mention_reliability": mention_reliability,
                        "feature_reliability": lam,
                        "weight": weight,
                        "contribution": contrib,
                        "support": stat["support"],
                    })

                elif spec.feature_type == "binary":
                    raw_val = row[feat_name]
                    is_present = _binary_present(raw_val, spec.active_values)
                    if not is_present:
                        continue
                    if True not in feat_stats:
                        continue

                    stat = feat_stats[True]
                    weight = stat["shrunk_weight"]
                    lam = spec.reliability
                    contrib = -1.0 * gsign * mention_reliability * lam * weight

                    mention_raw += contrib
                    mention_abs += abs(contrib)
                    evidence_total += mention_reliability * lam * abs(weight)

                    contribs.append({
                        "feature": feat_name,
                        "value": True,
                        "gender_sign": gsign,
                        "mention_reliability": mention_reliability,
                        "feature_reliability": lam,
                        "weight": weight,
                        "contribution": contrib,
                        "support": stat["support"],
                    })

            out = {
                "row_index": idx,
                self.config.item_id_col: row[self.config.item_id_col],
                self.config.gender_col: row[self.config.gender_col],
                "_gsign": gsign,
                "mention_reliability": mention_reliability,
                "mention_score_raw": mention_raw,
                "mention_abs_score": mention_abs,
                "mention_evidence_total": evidence_total,
                "contributions_json": json.dumps(contribs, ensure_ascii=False),
            }

            if self.config.mention_id_col and self.config.mention_id_col in row.index:
                out[self.config.mention_id_col] = row[self.config.mention_id_col]

            rows.append(out)

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # Item scoring
    # --------------------------------------------------------

    def score_items(self, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Score items from a target mention dataframe.

        Returns:
          item_scores_df, mention_scores_df
        """
        mention_scores = self.score_mentions(target_df)

        if mention_scores.empty:
            return (
                pd.DataFrame(columns=[
                    self.config.item_id_col,
                    "num_mentions",
                    "raw_score",
                    "normalized_score",
                    "final_score",
                    "intensity_score",
                    "feature_contributions_json",
                ]),
                mention_scores,
            )

        item_rows = []

        for item_id, grp in mention_scores.groupby(self.config.item_id_col, dropna=False):
            raw_score = float(grp["mention_score_raw"].sum())
            abs_score = float(grp["mention_abs_score"].sum())
            evidence_total = float(grp["mention_evidence_total"].sum())
            num_mentions = int(len(grp))

            if self.config.use_evidence_normalization:
                normalized = raw_score / (evidence_total + self.config.epsilon)
            else:
                normalized = raw_score / max(num_mentions, 1)

            final_score = math.tanh(self.config.tanh_gamma * normalized) if self.config.use_tanh else normalized
            intensity_score = abs_score / max(num_mentions, 1)

            # Aggregate feature-level contributions
            feature_totals: Dict[str, float] = {}
            feature_abs_totals: Dict[str, float] = {}

            for _, row in grp.iterrows():
                contribs = json.loads(row["contributions_json"])
                for c in contribs:
                    feat_key = f"{c['feature']}={c['value']}"
                    feature_totals[feat_key] = feature_totals.get(feat_key, 0.0) + float(c["contribution"])
                    feature_abs_totals[feat_key] = feature_abs_totals.get(feat_key, 0.0) + abs(float(c["contribution"]))

            sorted_features = sorted(
                [
                    {
                        "feature_value": k,
                        "net_contribution": v,
                        "abs_contribution": feature_abs_totals[k],
                    }
                    for k, v in feature_totals.items()
                ],
                key=lambda x: x["abs_contribution"],
                reverse=True,
            )

            item_rows.append({
                self.config.item_id_col: item_id,
                "num_mentions": num_mentions,
                "raw_score": raw_score,
                "normalized_score": normalized,
                "final_score": final_score,
                "intensity_score": intensity_score,
                "feature_contributions_json": json.dumps(sorted_features, ensure_ascii=False),
            })

        item_scores = pd.DataFrame(item_rows).sort_values(self.config.item_id_col).reset_index(drop=True)
        return item_scores, mention_scores
