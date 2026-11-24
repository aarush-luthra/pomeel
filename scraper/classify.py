"""Stage 4: Classify candidates against the CCPA 2023 and OECD definitions.

Backend (first available, or forced with --backend):
  - Nebius AI Studio, model NEBIUS_MODEL, when NEBIUS_API_KEY is set
  - OpenAI GPT-4o-mini when OPENAI_API_KEY is set
  - otherwise a transparent keyword/rule baseline, so the pipeline still runs

Every element gets: is_dark_pattern, pattern_type, confidence, reason.
Anything under config.CONFIDENCE_THRESHOLD is marked needs_review.

Before classifying, a countdown-consistency check runs: for each site, every
countdown seen is converted to an implied end time (snapshot time + time
remaining). If those end times disagree across page loads by more than a few
minutes, the timer is not counting down to a real deadline.
"""
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import config
import store

# ---------------------------------------------------------------- taxonomy
PATTERNS = {
    "false_urgency": ("False Urgency", "Urgency / Social proof"),
    "basket_sneaking": ("Basket Sneaking", "Sneaking"),
    "drip_pricing": ("Drip Pricing", "Sneaking"),
    "confirm_shaming": ("Confirm Shaming", "Interface interference"),
    "subscription_trap": ("Subscription Trap", "Obstruction"),
    "misleading_discount": ("Interface Interference (misleading price claim)", "Interface interference"),
    "interface_interference": ("Interface Interference", "Interface interference"),
}

SYSTEM_PROMPT = """You are a careful consumer-protection analyst. You label single UI elements
scraped from Indian e-commerce pages. Use these definitions.

India CCPA "Guidelines for Prevention and Regulation of Dark Patterns, 2023":
- false_urgency: falsely stating or implying urgency or scarcity to push an immediate purchase,
  including showing false popularity of a product (e.g. fake "only 2 left", "20 people viewing",
  countdown timers that reset and point to no real deadline).
- basket_sneaking: adding items, services, donations or charges to the cart without the user's
  consent at checkout, e.g. pre-ticked paid add-ons (insurance, gift wrap, donation, tip).
- drip_pricing: elements of price are not revealed upfront or are revealed surreptitiously later
  (e.g. convenience / platform / handling fees that first appear in the cart).
- confirm_shaming: using a phrase, video or design to create guilt or shame so the user complies
  (e.g. decline button "No, I don't want to save money").
- subscription_trap: hard or hidden cancellation, auto-renewal or payment details for a "free"
  trial without clear disclosure, cancel path not visible.
- interface_interference: design that highlights or obscures information to misdirect, including
  pre-selected consent boxes for marketing.
- misleading_discount: a discount claim that does not match the prices shown (e.g. "% off" badge
  larger than the real reduction from the struck-through price) or an implausibly inflated
  reference price.

OECD (2022) "Dark commercial patterns" categories: forced action, interface interference, nagging,
obstruction, sneaking, social proof, urgency.

Important: ordinary e-commerce content is NOT a dark pattern by default. A struck-through MRP with
an accurately computed discount, a free-shipping notice, a plain "No thanks" link, or a real sale
with a fixed end date are honest. Only label is_dark_pattern=true when the element itself shows the
manipulative property, and lower your confidence when you cannot verify a claim (e.g. you cannot
know if a stock count is true).

Answer ONLY with JSON:
{"is_dark_pattern": true|false,
 "pattern_type": one of ["false_urgency","basket_sneaking","drip_pricing","confirm_shaming",
                         "subscription_trap","interface_interference","misleading_discount","none"],
 "confidence": number between 0 and 1,
 "reason": "one sentence"}"""


# ---------------------------------------------------------------- countdown check
def _countdown_seconds(text):
    t = text.lower()
    m = re.search(r"(\d{1,2})\s*:\s*(\d{2})\s*:\s*(\d{2})", t)
    if m:
        h, mi, s = map(int, m.groups())
        d = re.search(r"(\d+)\s*d", t[: m.start()])
        return (int(d.group(1)) if d else 0) * 86400 + h * 3600 + mi * 60 + s
    parts = {}
    for unit, rx in (("d", r"(\d+)\s*(d|days?)\b"), ("h", r"(\d+)\s*(h|hrs?|hours?)\b"),
                     ("m", r"(\d+)\s*(m|mins?|minutes?)\b"), ("s", r"(\d+)\s*(s|secs?|seconds?)\b")):
        mm = re.search(rx, t)
        if mm:
            parts[unit] = int(mm.group(1))
    if "h" in parts or "m" in parts:
        return parts.get("d", 0) * 86400 + parts.get("h", 0) * 3600 + parts.get("m", 0) * 60 + parts.get("s", 0)
    return None


def countdown_consistency(cands):
    by_site = defaultdict(list)
    for c in cands:
        if c["element_type"] == "countdown":
            secs = _countdown_seconds(c["text"])
            if secs is not None:
                end = datetime.fromisoformat(c["timestamp"]) + timedelta(seconds=secs)
                by_site[c["site"]].append(end)
    out = {}
    for site, ends in by_site.items():
        if len(ends) < 2:
            out[site] = {"observations": len(ends), "spread_s": None, "verdict": "single observation"}
            continue
        spread = (max(ends) - min(ends)).total_seconds()
        out[site] = {
            "observations": len(ends), "spread_s": round(spread),
            "verdict": "consistent deadline" if spread <= 300 else "deadline shifts between page loads",
        }
    return out


# ---------------------------------------------------------------- keyword baseline
def _lbl(dark, ptype, conf, reason):
    return {"is_dark_pattern": dark, "pattern_type": ptype if dark else "none",
            "confidence": conf, "reason": reason}


def keyword_classify(c):
    t, et, x = c["text"].lower(), c["element_type"], c.get("extra", {})
    if et == "countdown":
        v = x.get("countdown_check", {}).get("verdict")
        if v == "deadline shifts between page loads":
            return _lbl(True, "false_urgency", 0.9,
                        f"Countdown's implied end time moved by {x['countdown_check']['spread_s']} s across page loads, so it is not tied to a real deadline.")
        if v == "consistent deadline":
            return _lbl(True, "false_urgency", 0.5,
                        "Countdown pressures an immediate purchase; its deadline was consistent across loads, so it may be a genuine sale end.")
        return _lbl(True, "false_urgency", 0.55, "Countdown timer creates time pressure; only one observation, so whether it resets is unverified.")
    if et == "urgency_scarcity":
        if re.search(r"only\s+\d+\s+(left|remaining)|few\s+\w+\s+left|almost\s+(gone|sold)|selling\s+(fast|out)|low\s+stock", t):
            return _lbl(True, "false_urgency", 0.65, "Scarcity claim pushes an immediate purchase; the stock level cannot be verified from the page.")
        return _lbl(True, "false_urgency", 0.55, "Generic urgency wording without a stated deadline or stock figure.")
    if et == "social_proof":
        return _lbl(True, "false_urgency", 0.6, "Activity/popularity message used to create urgency; the figure cannot be verified.")
    if et == "discount":
        gap = x.get("discount_gap_pct")
        comp = x.get("computed_discount_pct") or 0
        if gap is not None and abs(gap) > 3:
            return _lbl(True, "misleading_discount", 0.85,
                        f"Badge claims {x['claimed_discount_pct']}% off but the shown prices give {comp}%.")
        if comp >= 70:
            return _lbl(True, "misleading_discount", 0.5, f"Very large discount ({comp}%) may rest on an inflated reference price; needs checking.")
        return _lbl(False, None, 0.85, f"Struck-through price with an accurate discount ({comp}%); ordinary price display.")
    if et == "fee_line":
        if re.search(r"\bfree\b|₹\s*0(\.0+)?\b", t):
            return _lbl(False, None, 0.85, "Fee line shows a zero / free charge.")
        if c["page_type"] == "cart" and re.search(r"convenience|platform|handling|packaging|packing|processing|service|cod", t):
            return _lbl(True, "drip_pricing", 0.8, "A non-shipping fee appears on the cart page in addition to the product price.")
        if c["page_type"] == "cart":
            return _lbl(True, "drip_pricing", 0.6, "Shipping/delivery charge appears at cart; unclear if disclosed on the product page.")
        return _lbl(False, None, 0.6, "Fee is disclosed on a pre-cart page.")
    if et == "preselected_addon":
        if re.search(r"insur|protect|warranty|gift\s*wrap|donat|tip\b|charity|care\s*plan|extended", t):
            return _lbl(True, "basket_sneaking", 0.9, "A paid add-on is pre-ticked, so it is added unless the user unticks it.")
        return _lbl(True, "interface_interference", 0.75, "Marketing / communication consent is pre-ticked by default.")
    if et == "confirm_shaming":
        if re.search(r"don'?t\s+(want|like|need|care)|rather|full\s+price|miss\s+out|prefer\s+paying|i'?m\s+not", t):
            return _lbl(True, "confirm_shaming", 0.85, "Decline option is worded to make the user feel foolish for opting out.")
        return _lbl(False, None, 0.75, "Plain, neutral decline wording.")
    if et == "subscription":
        has_cancel = x.get("cancel_text_on_page")
        if re.search(r"auto[\s-]?renew|renews?\s+automatically|billed|recurring|free\s+trial", t):
            if has_cancel:
                return _lbl(False, None, 0.6, "Recurring terms shown and cancellation is mentioned on the page.")
            return _lbl(True, "subscription_trap", 0.65, "Recurring/trial terms with no visible cancellation path on the page.")
        return _lbl(False, None, 0.6, "Subscription/membership mention without hidden renewal terms.")
    return _lbl(False, None, 0.5, "No rule matched.")


# ---------------------------------------------------------------- LLM
def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def llm_classify(client, model, c):
    payload = {
        "element_type_from_rules": c["element_type"], "page_type": c["page_type"],
        "visible_text": c["text"], "html_snippet": c["snippet"][:1200], "extra": c.get("extra", {}),
    }
    kw = dict(model=model, temperature=0, max_tokens=300,
              messages=[{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    try:
        resp = client.chat.completions.create(response_format={"type": "json_object"}, **kw)
    except Exception as e:  # some hosted models don't support JSON mode; the prompt still asks for JSON
        if "response_format" not in str(e) and "json" not in str(e).lower():
            raise
        resp = client.chat.completions.create(**kw)
    out = _parse_json(resp.choices[0].message.content)
    out["pattern_type"] = out.get("pattern_type") or "none"
    out["reason"] = out.get("reason", "")
    out["confidence"] = float(out.get("confidence", 0))
    out["is_dark_pattern"] = bool(out.get("is_dark_pattern"))
    if not out["is_dark_pattern"]:
        out["pattern_type"] = "none"
    return out


# ---------------------------------------------------------------- run
def run(backend="auto"):
    cands = [json.loads(l) for l in config.CANDIDATES.read_text().splitlines() if l.strip()]
    checks = countdown_consistency(cands)
    for c in cands:
        if c["element_type"] == "countdown" and c["site"] in checks:
            c["extra"]["countdown_check"] = checks[c["site"]]

    if backend == "auto":
        backend = ("nebius" if os.environ.get("NEBIUS_API_KEY")
                   else "openai" if os.environ.get("OPENAI_API_KEY") else "keyword")
    use_llm = backend in ("nebius", "openai")
    client, model = None, None
    if use_llm:
        from openai import OpenAI
        if backend == "nebius":
            client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
            model = config.NEBIUS_MODEL
        else:
            client, model = OpenAI(), config.OPENAI_MODEL
    name = f"{backend}:{model}" if use_llm else "keyword-baseline"
    print(f"[classify] backend = {name}, {len(cands)} elements")

    for i, c in enumerate(cands, 1):
        try:
            lab = llm_classify(client, model, c) if use_llm else keyword_classify(c)
        except Exception as e:
            lab = keyword_classify(c)
            lab["reason"] += f" (LLM failed: {type(e).__name__}; keyword fallback used)"
        lab["needs_review"] = lab["confidence"] < config.CONFIDENCE_THRESHOLD
        lab["backend"] = name
        if lab["is_dark_pattern"] and lab["pattern_type"] in PATTERNS:
            lab["ccpa_label"], lab["oecd_category"] = PATTERNS[lab["pattern_type"]]
        else:
            lab["ccpa_label"], lab["oecd_category"] = "—", "—"
        c["label"] = lab
        if use_llm and i % 20 == 0:
            print(f"[classify] {i}/{len(cands)}", flush=True)

    with config.CLASSIFIED.open("w") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (config.DATA / "countdown_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    try:
        store.upsert(cands, with_labels=True)
    except Exception as e:
        print(f"[classify] warning: could not update ChromaDB labels ({e})")
    dark = sum(c["label"]["is_dark_pattern"] for c in cands)
    review = sum(c["label"]["needs_review"] for c in cands)
    print(f"[classify] {dark} flagged as dark patterns, {review} need review -> {config.CLASSIFIED.relative_to(config.ROOT)}")
    return cands
