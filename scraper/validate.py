"""Classifier validation.

`python run.py validate-sample` writes a stratified random sample (default 30)
of classified elements to data/validation_sample.csv. Open each crop image,
fill in `human_is_dark` (1/0) and optionally `human_pattern_type`, save, and
re-run `python run.py report`. The report then computes precision and the
other agreement figures from your labels.
"""
import csv
import json
import random
from collections import defaultdict

import config

FIELDS = ["id", "site", "sector", "page_type", "element_type", "text", "crop_path",
          "model_is_dark", "model_pattern_type", "model_confidence", "model_reason",
          "human_is_dark", "human_pattern_type", "notes"]


def make_sample(n=30, seed=42):
    cands = [json.loads(l) for l in config.CLASSIFIED.read_text().splitlines() if l.strip()]
    if config.VALIDATION_CSV.exists():
        print(f"[validate] {config.VALIDATION_CSV.name} already exists; delete it first to resample")
        return
    rng = random.Random(seed)
    # stratify by the model's label so both positives and negatives are checked,
    # weighted toward positives because precision is the headline number
    pos = [c for c in cands if c["label"]["is_dark_pattern"]]
    neg = [c for c in cands if not c["label"]["is_dark_pattern"]]
    k_pos = min(len(pos), max(n * 2 // 3, n - len(neg)))
    k_neg = min(len(neg), n - k_pos)
    # spread positives across pattern types
    by_type = defaultdict(list)
    for c in pos:
        by_type[c["label"]["pattern_type"]].append(c)
    for v in by_type.values():
        rng.shuffle(v)
    picked = []
    while len(picked) < k_pos and any(by_type.values()):
        for t in list(by_type):
            if by_type[t] and len(picked) < k_pos:
                picked.append(by_type[t].pop())
    picked += rng.sample(neg, k_neg)
    rng.shuffle(picked)
    with config.VALIDATION_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for c in picked:
            L = c["label"]
            w.writerow({"id": c["id"], "site": c["site"], "sector": c["sector"], "page_type": c["page_type"],
                        "element_type": c["element_type"], "text": c["text"], "crop_path": c.get("crop_path") or "",
                        "model_is_dark": int(L["is_dark_pattern"]), "model_pattern_type": L["pattern_type"],
                        "model_confidence": L["confidence"], "model_reason": L["reason"],
                        "human_is_dark": "", "human_pattern_type": "", "notes": ""})
    print(f"[validate] wrote {len(picked)} rows to {config.VALIDATION_CSV.relative_to(config.ROOT)} "
          f"({k_pos} model-positive, {k_neg} model-negative). Fill in human_is_dark (1/0).")


def scores():
    """Returns None if no human labels yet, else a dict of agreement metrics."""
    if not config.VALIDATION_CSV.exists():
        return None
    rows = list(csv.DictReader(config.VALIDATION_CSV.open()))
    lab = [r for r in rows if r["human_is_dark"].strip() in ("0", "1")]
    if not lab:
        return {"sample_size": len(rows), "labelled": 0}
    tp = sum(r["model_is_dark"] == "1" and r["human_is_dark"] == "1" for r in lab)
    fp = sum(r["model_is_dark"] == "1" and r["human_is_dark"] == "0" for r in lab)
    fn = sum(r["model_is_dark"] == "0" and r["human_is_dark"] == "1" for r in lab)
    tn = sum(r["model_is_dark"] == "0" and r["human_is_dark"] == "0" for r in lab)
    typed = [r for r in lab if r["model_is_dark"] == "1" and r["human_is_dark"] == "1" and r["human_pattern_type"].strip()]
    type_agree = sum(r["human_pattern_type"].strip() == r["model_pattern_type"] for r in typed)
    div = lambda a, b: round(a / b, 3) if b else None
    return {
        "sample_size": len(rows), "labelled": len(lab), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": div(tp, tp + fp), "recall_within_sample": div(tp, tp + fn),
        "accuracy": div(tp + tn, len(lab)),
        "pattern_type_agreement": div(type_agree, len(typed)), "pattern_type_n": len(typed),
    }
