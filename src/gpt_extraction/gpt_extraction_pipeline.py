from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI


# ============================================================
# 1. Schema
# ============================================================

GenderLabel = Literal["M", "F", "N", "andy", "unknown"]
MentionType = Literal[
    "proper", "pronoun", "common_noun", "kinship",
    "occupational", "group", "other_person"
]
GrammaticalRole = Literal["subject", "object", "possessive", "oblique", "other"]
DepPos = Literal["nsubj", "dobj", "iobj", "pobj", "poss", "appos", "conj", "root", "other"]
RootPos = Literal["VERB", "NOUN", "ADJ", "OTHER"]
Agency = Literal["high", "medium", "low", "none"]
Authority = Literal["high", "medium", "low", "none"]
ConjType = Literal["and", "or", "but", "while", "none"]
NumberType = Literal["singular", "plural", "unknown"]


class Mention(BaseModel):
    mention_id: str
    entity_id: str
    text: str
    start: int
    end: int

    mention_type: MentionType
    namedness: int
    kinship: int
    grammatical_role: GrammaticalRole
    dep_pos: DepPos
    root: str
    root_pos: RootPos
    agency: Agency
    authority: Authority
    generic_he: int
    conj: ConjType
    verb_negation: int
    number: NumberType

    personhood_confidence: float = Field(ge=0.0, le=1.0)
    coref_confidence: float = Field(ge=0.0, le=1.0)
    gender_evidence_local: List[str] = Field(default_factory=list)


class Entity(BaseModel):
    entity_id: str
    canonical_mention: str
    mention_ids: List[str]
    entity_type: Literal["person"] = "person"
    gender: GenderLabel
    gender_confidence: float = Field(ge=0.0, le=1.0)
    gender_evidence_chain: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    item_text: str
    mentions: List[Mention]
    entities: List[Entity]
    notes: List[str] = Field(default_factory=list)


# ============================================================
# 2. Prompting
# ============================================================

SYSTEM_PROMPT = """
You are an expert information extraction assistant for educational assessment materials.

Your task is to process one paragraph or item of educational text and produce structured annotation.

You must:
1. identify all person mentions in the text,
2. resolve coreference among person mentions,
3. infer gender for each coreference chain using only evidence in the text,
4. extract mention-level syntactic and semantic features relevant to gender-bias analysis.

A "person mention" includes:
- proper names of people,
- pronouns referring to people,
- person-denoting common nouns or noun phrases,
- kinship terms referring to people,
- occupational or role nouns referring to people,
- group references if they denote people (e.g., "the students", "scientists", "children"),
but exclude non-human entities, organizations, generic objects, and animals unless the text clearly treats them as people.

Coreference:
- Mentions belong to the same entity if they refer to the same person or same group of people in context.
- Output entity-level clusters.
- If a pronoun or noun phrase is ambiguous and cannot be resolved confidently, do not force a link.
  Put it in a singleton entity and explain in notes.

Gender inference:
Infer gender at the entity level using only textual evidence from:
- pronouns (he, she, his, her, etc.),
- gendered kinship terms (mother, father, son, daughter, etc.),
- gendered role nouns (actress, waitress, prince, princess, etc.),
- titles if clearly gendered in context (Mr., Mrs., etc.),
- proper names only if the name is strongly associated with a gender and there is no contradictory evidence.

Do NOT use stereotypes or world knowledge beyond common lexical gender cues.
Do NOT infer gender from occupations alone.
Do NOT infer gender from activities, household roles, power, authority, or social stereotypes.

Allowed gender labels:
- M
- F
- N
- andy
- unknown

Definitions:
- M = male
- F = female
- N = neutral / explicitly non-gendered / plural mixed or neutral where appropriate
- andy = androgynous / genuinely ambiguous between male and female
- unknown = insufficient evidence

If mentions in the same coreference chain conflict:
- prefer explicit pronoun evidence over name-based inference,
- prefer lexical gender terms over name-based inference,
- if conflict remains unresolved, assign andy and explain.

Extract the following mention-level features:
1. mention_type: proper | pronoun | common_noun | kinship | occupational | group | other_person
2. namedness: 1 if proper name, else 0
3. kinship: 1 if kinship term or contains one, else 0
4. grammatical_role: subject | object | possessive | oblique | other
5. dep_pos: nsubj | dobj | iobj | pobj | poss | appos | conj | root | other
6. root: main associated predicate or lexical head
7. root_pos: VERB | NOUN | ADJ | OTHER
8. agency: high | medium | low | none
9. authority: high | medium | low | none
10. generic_he: 1 if masculine form is generic, else 0
11. conj: and | or | but | while | none
12. verb_negation: 1 if main associated predicate is negated, else 0
13. number: singular | plural | unknown
14. personhood_confidence: float in [0,1]
15. coref_confidence: float in [0,1]

Output requirements:
- Return structured data only.
- Preserve exact mention text as it appears in the input.
- Provide character start and end offsets using Python-style indexing.
- Mentions should not overlap unless necessary for true apposition or nested reference.
- Every mention must belong to exactly one entity_id.
- Every entity must list all of its mention_ids.
- Sort mentions by start offset.
- Use stable IDs: M1, M2, ... and E1, E2, ...
""".strip()


def build_messages(item_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
Process the following educational assessment item.

Instructions:
- Extract all person mentions.
- Resolve coreference among person mentions.
- Infer gender at the entity/coreference-chain level.
- Extract the required mention-level features.
- Use only evidence in the text.
- Follow this internal order:
  (1) identify person-denoting spans,
  (2) cluster them into entities,
  (3) assign entity-level gender,
  (4) derive mention-level features,
  (5) validate offsets and IDs.

Text:
{item_text}
""".strip(),
        },
    ]


# ============================================================
# 3. Logging helpers
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def preview_text(text: str, n: int = 100) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


# ============================================================
# 4. Validation
# ============================================================

import re
from typing import Optional, Tuple


def normalize_span_text(s: str) -> str:
    """
    Light normalization for matching mention text to source text.
    """
    if s is None:
        return ""
    return (
        s.replace("\u2018", "'")
         .replace("\u2019", "'")
         .replace("\u201c", '"')
         .replace("\u201d", '"')
         .replace("\u2013", "-")
         .replace("\u2014", "-")
    )


def trim_whitespace_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    Trim leading/trailing whitespace from a span.
    """
    start = max(0, start)
    end = min(len(text), end)

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def recover_best_span(
    full_text: str,
    mention_text: str,
    approx_start: int,
    search_radius: int = 40,
) -> Tuple[int, int]:
    """
    Try to recover the best span for mention_text near approx_start.

    Returns:
        (start, end), or (-1, -1) if not found.
    """
    if not mention_text:
        return -1, -1

    full_text_norm = normalize_span_text(full_text)
    mention_text_norm = normalize_span_text(mention_text)

    approx_start = max(0, min(len(full_text_norm), approx_start))
    left = max(0, approx_start - search_radius)
    right = min(len(full_text_norm), approx_start + search_radius + len(mention_text_norm))

    window = full_text_norm[left:right]

    # 1) exact local search
    idx = window.find(mention_text_norm)
    if idx != -1:
        s = left + idx
        e = s + len(mention_text_norm)
        return s, e

    # 2) case-insensitive local search
    idx = window.lower().find(mention_text_norm.lower())
    if idx != -1:
        s = left + idx
        e = s + len(mention_text_norm)
        return s, e

    return -1, -1


def auto_fix_mention_offsets(result: ExtractionResult) -> List[str]:
    """
    Mutates result in place by repairing mention offsets when possible.

    Returns:
        list of fix messages
    """
    fixes: List[str] = []
    text = result.item_text

    for m in result.mentions:
        # Skip obviously impossible spans first
        if not (0 <= m.start <= m.end <= len(text)):
            old_start, old_end = m.start, m.end
            new_start, new_end = recover_best_span(text, m.text, max(0, m.start))
            if new_start != -1:
                m.start, m.end = new_start, new_end
                fixes.append(
                    f"{m.mention_id}: repaired invalid span ({old_start}, {old_end}) -> ({new_start}, {new_end})"
                )
            continue

        extracted = text[m.start:m.end]
        if extracted == m.text:
            continue

        # 1) try whitespace trim on existing span
        t_start, t_end = trim_whitespace_span(text, m.start, m.end)
        trimmed = text[t_start:t_end]
        if trimmed == m.text:
            old_start, old_end = m.start, m.end
            m.start, m.end = t_start, t_end
            fixes.append(
                f"{m.mention_id}: trimmed whitespace span ({old_start}, {old_end}) -> ({t_start}, {t_end})"
            )
            continue

        # 2) try nearby recovery using model text
        old_start, old_end = m.start, m.end
        new_start, new_end = recover_best_span(text, m.text, m.start)
        if new_start != -1:
            m.start, m.end = new_start, new_end
            fixes.append(
                f"{m.mention_id}: recovered span ({old_start}, {old_end}) -> ({new_start}, {new_end})"
            )
            continue

    return fixes


def validate_result(result: ExtractionResult) -> List[str]:
    errors: List[str] = []
    text = result.item_text

    mention_ids = set()
    entity_ids = set()

    for m in result.mentions:
        mention_ids.add(m.mention_id)

        if not (0 <= m.start <= m.end <= len(text)):
            errors.append(
                f"{m.mention_id}: invalid offsets ({m.start}, {m.end}) for text length {len(text)}"
            )
            continue

        extracted = text[m.start:m.end]
        if extracted != m.text:
            errors.append(
                f"{m.mention_id}: offsets yield {extracted!r} but mention text is {m.text!r}"
            )

    for e in result.entities:
        entity_ids.add(e.entity_id)

    for m in result.mentions:
        if m.entity_id not in entity_ids:
            errors.append(f"{m.mention_id}: unknown entity_id {m.entity_id}")

    for e in result.entities:
        for mid in e.mention_ids:
            if mid not in mention_ids:
                errors.append(f"{e.entity_id}: unknown mention_id {mid}")

    if len(mention_ids) != len(result.mentions):
        errors.append("duplicate mention_id found")

    if len(entity_ids) != len(result.entities):
        errors.append("duplicate entity_id found")

    starts = [m.start for m in result.mentions]
    if starts != sorted(starts):
        errors.append("mentions are not sorted by start offset")

    return errors


def validate_mentions_have_gender(result: ExtractionResult) -> List[str]:
    errors: List[str] = []
    entity_map = {e.entity_id: e for e in result.entities}

    for m in result.mentions:
        if m.entity_id not in entity_map:
            errors.append(f"{m.mention_id}: missing entity {m.entity_id}")
            continue

        g = entity_map[m.entity_id].gender
        if g not in {"M", "F", "N", "andy", "unknown"}:
            errors.append(f"{m.mention_id}: invalid propagated gender {g!r}")

    return errors


# ============================================================
# 5. Extraction call
# ============================================================

def extract_item_with_llm(
    item_text: str,
    client: OpenAI,
    model: str,
) -> ExtractionResult:
    response = client.responses.parse(
        model=model,
        input=build_messages(item_text),
        text_format=ExtractionResult,
    )

    if getattr(response, "status", None) == "incomplete":
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", None) or "unknown"
        raise RuntimeError(f"Incomplete response: {reason}")

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise RuntimeError(
            f"No parsed output returned. Response status={getattr(response, 'status', None)!r}"
        )

    return parsed


# ============================================================
# 6. Flatten helpers
# ============================================================

def flatten_mentions(item_id: str, result: ExtractionResult) -> List[dict]:
    rows = []

    entity_map = {e.entity_id: e for e in result.entities}

    for m in result.mentions:
        ent = entity_map.get(m.entity_id, None)

        gender = ent.gender if ent is not None else "unknown"
        gender_confidence = ent.gender_confidence if ent is not None else None
        gender_evidence_chain = ent.gender_evidence_chain if ent is not None else []

        rows.append({
            "item_id": item_id,
            "mention_id": m.mention_id,
            "entity_id": m.entity_id,

            "gender": gender,
            "mention_reliability": m.coref_confidence,

            "gender_confidence": gender_confidence,
            "gender_evidence_chain": json.dumps(gender_evidence_chain, ensure_ascii=False),

            "item_text": result.item_text,
            "mention_text": m.text,
            "start": m.start,
            "end": m.end,

            "mention_type": m.mention_type,
            "namedness": m.namedness,
            "kinship": m.kinship,
            "grammatical_role": m.grammatical_role,
            "dep_pos": m.dep_pos,
            "root": m.root,
            "root_pos": m.root_pos,
            "agency": m.agency,
            "authority": m.authority,
            "generic_he": m.generic_he,
            "conj": m.conj,
            "verb_negation": m.verb_negation,
            "number": m.number,

            "personhood_confidence": m.personhood_confidence,
            "coref_confidence": m.coref_confidence,
            "gender_evidence_local": json.dumps(m.gender_evidence_local, ensure_ascii=False),
        })

    return rows


def flatten_entities(item_id: str, result: ExtractionResult) -> List[dict]:
    rows = []
    for e in result.entities:
        rows.append({
            "item_id": item_id,
            "entity_id": e.entity_id,
            "canonical_mention": e.canonical_mention,
            "mention_ids": json.dumps(e.mention_ids, ensure_ascii=False),
            "entity_type": e.entity_type,
            "gender": e.gender,
            "gender_confidence": e.gender_confidence,
            "gender_evidence_chain": json.dumps(e.gender_evidence_chain, ensure_ascii=False),
        })
    return rows


# ============================================================
# 7. Batch runner
# ============================================================

def run_batch(
    input_csv: Path,
    output_dir: Path,
    model: str,
    item_id_col: str,
    text_col: str,
    sleep_seconds: float = 0.0,
    max_items: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[START] Loading input CSV from: {input_csv}")
    df = pd.read_csv(input_csv)
    log(f"[INFO] Loaded {len(df)} rows")

    if max_items is not None:
        df = df.head(max_items).copy()
        log(f"[INFO] Restricted to first {len(df)} rows because max_items={max_items}")

    if item_id_col not in df.columns:
        raise ValueError(f"Missing required column: {item_id_col}")
    if text_col not in df.columns:
        raise ValueError(f"Missing required column: {text_col}")

    client = OpenAI()

    jsonl_path = output_dir / "extractions.jsonl"
    mentions_csv_path = output_dir / "mentions.csv"
    entities_csv_path = output_dir / "entities.csv"
    errors_csv_path = output_dir / "errors.csv"

    mention_rows: List[dict] = []
    entity_rows: List[dict] = []
    error_rows: List[dict] = []

    total = len(df)
    start_batch_time = time.time()

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for idx, row in df.iterrows():
            item_num = idx + 1
            item_id = str(row[item_id_col])
            item_text = str(row[text_col])

            log("=" * 80)
            log(f"[PROCESS] Item {item_num}/{total} | item_id={item_id}")
            log(f"[TEXT] {preview_text(item_text, 140)}")

            item_start = time.time()

            try:
                log(f"[LLM] Sending request for item_id={item_id} ...")
                result = extract_item_with_llm(
                    item_text=item_text,
                    client=client,
                    model=model,
                )

                # auto-fix offsets before validation
                fix_messages = auto_fix_mention_offsets(result)
                if fix_messages:
                    log(f"[AUTOFIX] item_id={item_id} repaired {len(fix_messages)} span issue(s)")
                    for msg in fix_messages[:5]:
                        log(f"  - {msg}")
                    if len(fix_messages) > 5:
                        log(f"  ... and {len(fix_messages) - 5} more")

                validation_errors = validate_result(result)
                validation_errors.extend(validate_mentions_have_gender(result))

                if validation_errors:
                    log(f"[VALIDATION] item_id={item_id} had {len(validation_errors)} issue(s)")
                    for err in validation_errors[:5]:
                        log(f"  - {err}")
                    if len(validation_errors) > 5:
                        log(f"  ... and {len(validation_errors) - 5} more")
                else:
                    log(f"[VALIDATION] item_id={item_id} passed")

                record = {
                    "item_id": item_id,
                    "item_text": result.item_text,
                    "mentions": [m.model_dump() for m in result.mentions],
                    "entities": [e.model_dump() for e in result.entities],
                    "notes": result.notes,
                    "autofix_messages": fix_messages,
                    "validation_errors": validation_errors,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                mention_rows.extend(flatten_mentions(item_id, result))
                entity_rows.extend(flatten_entities(item_id, result))

                if validation_errors:
                    error_rows.append({
                        "item_id": item_id,
                        "row_index": idx,
                        "error_type": "validation",
                        "message": " | ".join(validation_errors),
                    })

                total_item_time = time.time() - item_start
                log(f"[DONE] item_id={item_id} finished in {total_item_time:.2f}s")

            except Exception as e:
                total_item_time = time.time() - item_start
                log(f"[ERROR] item_id={item_id} failed after {total_item_time:.2f}s")
                log(f"[ERROR] {type(e).__name__}: {e}")

                error_rows.append({
                    "item_id": item_id,
                    "row_index": idx,
                    "error_type": "exception",
                    "message": str(e),
                })

            if sleep_seconds > 0:
                log(f"[WAIT] Sleeping for {sleep_seconds:.2f}s")
                time.sleep(sleep_seconds)

    pd.DataFrame(mention_rows).to_csv(mentions_csv_path, index=False)
    pd.DataFrame(entity_rows).to_csv(entities_csv_path, index=False)
    pd.DataFrame(error_rows).to_csv(errors_csv_path, index=False)

    elapsed = time.time() - start_batch_time
    log("=" * 80)
    log(f"[FINISH] Batch complete in {elapsed:.2f}s")
    log(f"[OUTPUT] JSONL:    {jsonl_path}")
    log(f"[OUTPUT] Mentions: {mentions_csv_path}")
    log(f"[OUTPUT] Entities: {entities_csv_path}")
    log(f"[OUTPUT] Errors:   {errors_csv_path}")
    log(f"[SUMMARY] mention_rows={len(mention_rows)} entity_rows={len(entity_rows)} error_rows={len(error_rows)}")


# ============================================================
# 8. CLI / notebook entry
# ============================================================

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract person mentions, coreference, gender, and features from educational items."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--item_id_col", type=str, default="item_id")
    parser.add_argument("--text_col", type=str, default="item_text")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--max_items", type=int, default=None)
    return parser.parse_args(args)


def main(cli_args=None) -> None:
    args = parse_args(cli_args)
    run_batch(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        model=args.model,
        item_id_col=args.item_id_col,
        text_col=args.text_col,
        sleep_seconds=args.sleep_seconds,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        print("Notebook detected. Use run_batch(...) directly or call:")
        print(
            'main(["--input_csv", "/path/to/input.csv", "--output_dir", "/path/to/output_dir"])'
        )
    else:
        main()
