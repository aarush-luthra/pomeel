"""Dark-pattern detection pipeline — command line entry point.

    python run.py all                 # collect -> extract -> store -> classify -> report
    python run.py collect [--site Plum --site Noise] [--headed]
    python run.py extract
    python run.py store
    python run.py classify [--backend auto|nebius|openai|keyword]
    python run.py validate-sample     # draw 30 elements for hand-labelling
    python run.py report
    python run.py query "only a few left in stock" [-n 10]
"""
import argparse

import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["all", "collect", "extract", "store", "classify",
                                      "validate-sample", "report", "query"])
    ap.add_argument("text", nargs="?", help="query text (for `query`)")
    ap.add_argument("--site", action="append", help="limit collection to these site names")
    ap.add_argument("--headed", action="store_true", help="show the browser while collecting")
    ap.add_argument("--backend", default="auto", choices=["auto", "nebius", "openai", "keyword"])
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()

    if a.stage in ("all", "collect"):
        import collect
        collect.run(only=a.site, headless=not a.headed)
    if a.stage in ("all", "extract"):
        import extract
        extract.run()
    if a.stage in ("all", "store"):
        import store
        try:
            store.run()
        except Exception as e:  # embeddings need a one-time model download
            print(f"[store] skipped: {e}")
    if a.stage in ("all", "classify"):
        import classify
        classify.run(a.backend)
    if a.stage == "validate-sample":
        import validate
        validate.make_sample(a.n if a.n != 10 else 30)
    if a.stage in ("all", "report"):
        import report
        report.run()
    if a.stage == "query":
        import store
        store.query(a.text or "urgency message", n=a.n)


if __name__ == "__main__":
    main()
