"""Stage 2: Extract candidate elements.

A rules layer over the saved, annotated HTML (parsed with BeautifulSoup) that
pulls out anything that *might* be a dark pattern. Every candidate keeps its
HTML snippet, its text, and a crop of the page screenshot. Whether it really
is a dark pattern is decided later by the classifier (Stage 4).
"""
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

import config

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- rules
RX = lambda *ps: [re.compile(p, re.I) for p in ps]

URGENCY = RX(
    r"\bonly\s+\d+\s+(left|remaining|in\s+stock|pieces?|units?|items?)\b",
    r"\b(few|limited)\s+(items?|pieces?|stocks?|units?|quantity)\s+(left|remaining)\b",
    r"\bhurry\b", r"\bselling\s+(fast|out)\b", r"\balmost\s+(gone|sold\s*out)\b",
    r"\blow\s+(in\s+)?stock\b", r"\blast\s+(few|chance)\b", r"\bhigh\s+demand\b",
    r"\blimited[\s-]+(time|period|stock)\b", r"\b(deal|offer|sale)\s+ends?\b", r"\bends\s+in\b",
)
COUNTDOWN = RX(
    r"\b\d{1,2}\s*:\s*\d{2}\s*:\s*\d{2}\b",
    r"\b\d+\s*(h|hrs?|hours?)\b\W{0,4}\d+\s*(m|mins?|minutes?)\b",
    r"\b(hrs?|hours?)\s+(mins?|minutes?)\s+(secs?|seconds?)\b",
)
SOCIAL_PROOF = RX(
    r"\b\d[\d,]*\+?\s+(people|others|users|customers|shoppers)\b.{0,30}\b(viewing|watching|looking|bought|purchased|added|ordered)\b",
    r"\b(in|added\s+to)\s+\d+\s+(carts?|bags?)\b",
    r"\b\d[\d,.]*\s*k?\+?\s+(bought|sold|purchased|ordered)\b.{0,25}\b(past|last|today|this)\b",
    r"\bsomeone\s+(in|from)\s+\w+.{0,30}\b(bought|purchased|ordered)\b",
)
FEE = RX(
    r"\b(convenience|handling|platform|packaging|packing|processing|service|gift\s*wrap|cod|cash\s+on\s+delivery|shipping|delivery)\s*(fee|fees|charges?|cost)\b",
)
CONFIRM_SHAME = RX(
    r"^\s*no,?\s+thanks?\b.{0,60}$",
    r"\bi\s+(don'?t|do\s+not)\s+(want|like|need|care)\b",
    r"\bi'?d\s+rather\b", r"\bno,?\s+i\s+(prefer|like|want|'?ll)\b",
    r"\bi\s+(prefer|like|love)\s+(paying|full\s+price)\b", r"\bmiss\s+out\b",
    r"\bno,?\s+i'?m\s+(good|fine|not)\b",
)
SUBSCRIPTION = RX(
    r"\bfree\s+trial\b", r"\bauto[\s-]?renew", r"\brenews?\s+automatically\b",
    r"\bsubscribe\s*(&|and)\s*save\b", r"\brecurring\b", r"\bbilled\s+(monthly|annually|every|yearly)\b",
    r"\bcancel\s+any\s*time\b", r"\bmembership\b",
)
ADDON_WORDS = re.compile(
    r"insur|protect|warranty|gift\s*wrap|donat|tip\b|newsletter|subscribe|marketing|offers|"
    r"updates|whatsapp|sms|promotional|charity|care\s*plan|extended", re.I)

PRICE = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)", re.I)
PCT_OFF = re.compile(r"(\d{1,2})\s*%\s*off", re.I)

MAX_TEXT = 200
MAX_DUPES = 3  # keep at most N copies of identical text per page/rule


# ---------------------------------------------------------------- helpers
def bbox_of(el):
    while el is not None and getattr(el, "attrs", None) is not None:
        b = el.attrs.get("data-dp-bbox")
        if b:
            try:
                return [int(v) for v in b.split(",")]
            except ValueError:  # e.g. "NaN" from an element with no layout box
                return None
        el = el.parent
    return None


def clean_snippet(el, limit=700):
    s = str(el)
    s = re.sub(r'\sdata-dp-[a-z]+="[^"]*"', "", s)
    s = re.sub(r"\s+", " ", s)
    return s[:limit]


def text_of(el):
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))


def prices_in(text):
    out = []
    for m in PRICE.finditer(text):
        if re.match(r"\s*/\s*\d*\s*(g|gm|kg|ml|l|unit|pc|piece|100)", text[m.end():], re.I):
            continue  # unit price such as "₹9.88 / g", not a selling price
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


def smallest_matches(soup, patterns):
    """Elements whose text matches, keeping only the innermost match."""
    hits = []
    for el in soup.body.find_all(True) if soup.body else []:
        if el.name in ("script", "style", "noscript", "svg") or "data-dp-bbox" not in el.attrs:
            continue
        t = text_of(el)
        if not t or len(t) > MAX_TEXT:
            continue
        if any(p.search(t) for p in patterns):
            hits.append(el)
    hit_ids = {id(h) for h in hits}
    return [h for h in hits if not any(id(d) in hit_ids for d in h.find_all(True))]


def crop(screenshot, bbox, out_path, pad=40):
    try:
        with Image.open(screenshot) as im:
            x, y, w, h = bbox
            box = (max(0, x - pad), max(0, y - pad),
                   min(im.width, x + w + pad), min(im.height, y + h + pad))
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            im.crop(box).convert("RGB").save(out_path, "JPEG", quality=85)
            return str(out_path.relative_to(config.ROOT))
    except Exception:
        return None


# ---------------------------------------------------------------- extractors
def ex_text_rule(soup, element_type, patterns):
    for el in smallest_matches(soup, patterns):
        yield el, element_type, {}


def ex_discounts(soup):
    struck = [el for el in soup.find_all(True)
              if (el.name in ("s", "del", "strike") or el.attrs.get("data-dp-strike") == "1")
              and "data-dp-bbox" in el.attrs and PRICE.search(text_of(el))]
    for el in struck:
        mrp = prices_in(text_of(el))
        if not mrp:
            continue
        mrp = mrp[0]
        container = el
        for _ in range(4):  # walk up until the block holds a second (sale) price
            if container.parent is None:
                break
            container = container.parent
            t = text_of(container)
            if len(t) > 300:
                break
            others = [p for p in prices_in(t) if p != mrp]
            if others:
                sale = min(others)
                if sale >= mrp or sale <= 0:
                    break
                computed = round((mrp - sale) / mrp * 100, 1)
                claimed = PCT_OFF.search(t)
                claimed = int(claimed.group(1)) if claimed else None
                yield container, "discount", {
                    "mrp": mrp, "sale_price": sale, "computed_discount_pct": computed,
                    "claimed_discount_pct": claimed,
                    "discount_gap_pct": None if claimed is None else round(claimed - computed, 1),
                }
                break


def ex_preselected(soup):
    for inp in soup.find_all("input"):
        if inp.get("type") not in ("checkbox", "radio"):
            continue
        checked = inp.attrs.get("data-dp-checked") == "1" or (
            "data-dp-checked" not in inp.attrs and inp.has_attr("checked"))
        if not checked:
            continue
        label = None
        if inp.get("id"):
            label = soup.find("label", attrs={"for": inp["id"]})
        holder = label or inp
        node = holder
        for _ in range(3):
            if len(text_of(node)) >= 4 or node.parent is None:
                break
            node = node.parent
        t = text_of(node)
        if t and len(t) <= MAX_TEXT and ADDON_WORDS.search(t):
            yield node, "preselected_addon", {"input_type": inp.get("type")}


# ---------------------------------------------------------------- main
def extract_page(meta):
    html = (config.ROOT / meta["html_path"]).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    page_text = text_of(soup.body) if soup.body else ""
    shot = config.ROOT / meta["screenshot_path"]

    generators = [
        ex_text_rule(soup, "urgency_scarcity", URGENCY),
        ex_text_rule(soup, "countdown", COUNTDOWN),
        ex_text_rule(soup, "social_proof", SOCIAL_PROOF),
        ex_text_rule(soup, "fee_line", FEE),
        ex_text_rule(soup, "confirm_shaming", CONFIRM_SHAME),
        ex_text_rule(soup, "subscription", SUBSCRIPTION),
        ex_discounts(soup),
        ex_preselected(soup),
    ]
    seen, out = {}, []
    for gen in generators:
        for el, etype, extra in gen:
            text = text_of(el)
            if etype == "fee_line" and not (PRICE.search(text) or re.search(r"\bfree\b", text, re.I)):
                continue  # a fee mention without an amount is not a fee line
            key = (etype, text.lower())
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > MAX_DUPES:
                continue
            bbox = bbox_of(el)
            rid = hashlib.sha1(f"{meta['site']}|{meta['page_type']}|{meta['url']}|{etype}|{text}|{seen[key]}".encode()).hexdigest()[:12]
            if etype == "subscription":
                extra = {**extra, "cancel_text_on_page": bool(re.search(r"\bcancel", page_text, re.I))}
            out.append({
                "id": rid,
                "site": meta["site"], "sector": meta["sector"], "page_type": meta["page_type"],
                "url": meta["url"], "timestamp": meta["timestamp"],
                "element_type": etype, "text": text, "snippet": clean_snippet(el),
                "bbox": bbox, "screenshot_path": meta["screenshot_path"],
                "crop_path": None, "extra": extra,
            })
    # cap discount candidates per page (category pages list dozens of products);
    # keep mismatches first since those are the interesting ones
    disc = [c for c in out if c["element_type"] == "discount"]
    if len(disc) > 6:
        disc.sort(key=lambda c: -abs(c["extra"].get("discount_gap_pct") or 0))
        drop = {c["id"] for c in disc[6:]}
        out = [c for c in out if c["id"] not in drop]
    for c in out:
        if c["bbox"]:
            c["crop_path"] = crop(shot, c["bbox"], config.CROPS / f"{c['id']}.jpg")
    for c in out:
        c["page_dup_count"] = seen[(c["element_type"], c["text"].lower())]
    return out


def run():
    config.CROPS.mkdir(parents=True, exist_ok=True)
    all_c = []
    for meta_file in sorted(config.RAW.glob("*/meta.json")):
        site_meta = json.loads(meta_file.read_text())
        for page in site_meta["pages"]:
            cands = extract_page(page)
            all_c.extend(cands)
            print(f"[extract] {page['site']:<18} {page['page_type']:<9} {len(cands):>3} candidates", flush=True)
    with config.CANDIDATES.open("w") as f:
        for c in all_c:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[extract] total {len(all_c)} candidates -> {config.CANDIDATES.relative_to(config.ROOT)}")
    return all_c
