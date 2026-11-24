"""Stage 3: Store candidates in ChromaDB.

One record per candidate element: the visible text is embedded, the HTML
snippet is kept as the document body, and site / sector / page type / element
type / screenshot path / timestamp go in metadata. This also makes semantic
queries possible, e.g. `python run.py query "only a few items left in stock"`.
"""
import json

import chromadb

import config

COLLECTION = "dark_pattern_candidates"


def _client():
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _meta(c, label=None):
    m = {
        "site": c["site"], "sector": c["sector"], "page_type": c["page_type"],
        "element_type": c["element_type"], "url": c["url"], "timestamp": c["timestamp"],
        "screenshot_path": c["screenshot_path"], "crop_path": c.get("crop_path") or "",
        "snippet": c["snippet"][:2000],
    }
    if label:
        m.update({
            "is_dark_pattern": bool(label["is_dark_pattern"]),
            "pattern_type": label["pattern_type"] or "none",
            "confidence": float(label["confidence"]),
            "needs_review": bool(label["needs_review"]),
        })
    return m


def upsert(records, with_labels=False):
    col = _client().get_or_create_collection(COLLECTION)
    if not records:
        return 0
    col.upsert(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[_meta(r, r.get("label") if with_labels else None) for r in records],
    )
    return col.count()


def run():
    recs = [json.loads(l) for l in config.CANDIDATES.read_text().splitlines() if l.strip()]
    n = upsert(recs)
    print(f"[store] {len(recs)} candidates upserted; collection now holds {n}")


def query(text, n=10, where=None):
    col = _client().get_or_create_collection(COLLECTION)
    res = col.query(query_texts=[text], n_results=n, where=where)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        tag = meta.get("pattern_type", "unclassified")
        print(f"{dist:.3f}  [{meta['site']} / {meta['page_type']} / {meta['element_type']} -> {tag}]  {doc[:100]}")
