"""Stages 5 + 6: Analyze and write the findings report (report/report.html).

Descriptive statistics only. With ~12 sites, inferential tests across sectors
(e.g. chi-square) would be meaningless; they are left for the full 50-site run.

Counting rule: an element "counts" as a dark pattern when the classifier says
is_dark_pattern AND confidence >= threshold. Lower-confidence flags are shown
separately as "needs review" and never folded into the headline numbers.
"""
import html
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import config
import validate
from classify import PATTERNS

# palette (dataviz reference instance)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

SECTORS = ["Electronics", "Fashion", "Cosmetics"]
ORDER = ["false_urgency", "misleading_discount", "drip_pricing", "basket_sneaking",
         "confirm_shaming", "subscription_trap", "interface_interference"]
SHORT = {"false_urgency": "False urgency", "misleading_discount": "Misleading discount",
         "drip_pricing": "Drip pricing", "basket_sneaking": "Basket sneaking",
         "confirm_shaming": "Confirm shaming", "subscription_trap": "Subscription trap",
         "interface_interference": "Interface interference"}

e = html.escape


def load():
    cands = [json.loads(l) for l in config.CLASSIFIED.read_text().splitlines() if l.strip()]
    metas = [json.loads(p.read_text()) for p in sorted(config.RAW.glob("*/meta.json"))]
    return cands, metas


def counts_dp(c):
    L = c["label"]
    return L["is_dark_pattern"] and not L["needs_review"]


def review_dp(c):
    L = c["label"]
    return L["is_dark_pattern"] and L["needs_review"]


# ---------------------------------------------------------------- analysis
def analyze(cands, metas):
    sites = [m["site"] for m in metas if m["pages"]]
    sector_of = {m["site"]: m["sector"] for m in metas}
    per_site = {s: Counter() for s in sites}
    per_site_review = {s: Counter() for s in sites}
    for c in cands:
        if c["site"] not in per_site:
            continue
        if counts_dp(c):
            per_site[c["site"]][c["label"]["pattern_type"]] += 1
        elif review_dp(c):
            per_site_review[c["site"]][c["label"]["pattern_type"]] += 1

    with_dp = [s for s in sites if per_site[s]]
    with_any = [s for s in sites if per_site[s] or per_site_review[s]]
    prevalence = {"Overall": (len(with_dp), len(sites), len(with_any))}
    for sec in SECTORS:
        ss = [s for s in sites if sector_of[s] == sec]
        prevalence[sec] = (sum(1 for s in ss if per_site[s]), len(ss),
                           sum(1 for s in ss if per_site[s] or per_site_review[s]))

    freq_elements = Counter()
    freq_review = Counter()
    freq_sites = Counter()
    for s in sites:
        freq_elements.update(per_site[s])
        freq_review.update(per_site_review[s])
        for p in per_site[s]:
            freq_sites[p] += 1

    matrix = {sec: Counter() for sec in SECTORS}
    matrix_review = {sec: Counter() for sec in SECTORS}
    sector_n = Counter(sector_of[s] for s in sites)
    for s in sites:
        for p in per_site[s]:
            matrix[sector_of[s]][p] += 1
        for p in per_site_review[s]:
            if p not in per_site[s]:
                matrix_review[sector_of[s]][p] += 1

    n = len(sites) or 1
    return {
        "sites": sites, "sector_of": sector_of, "per_site": per_site, "per_site_review": per_site_review,
        "prevalence": prevalence, "freq_elements": freq_elements, "freq_review": freq_review,
        "freq_sites": freq_sites, "matrix": matrix, "matrix_review": matrix_review, "sector_n": sector_n,
        "avg_types_per_site": round(sum(len(per_site[s]) for s in sites) / n, 2),
        "avg_elements_per_site": round(sum(sum(per_site[s].values()) for s in sites) / n, 2),
    }


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_frequency(A, path):
    types = [p for p in ORDER if A["freq_elements"][p] or A["freq_review"][p]] or ORDER[:3]
    conf = [A["freq_elements"][p] for p in types]
    rev = [A["freq_review"][p] for p in types]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(types) + 1.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = range(len(types))[::-1]
    ax.barh(list(y), conf, color=BLUE, height=0.55, label="Counted (confidence ≥ threshold)",
            edgecolor=SURFACE, linewidth=2)
    ax.barh(list(y), rev, left=conf, color=ORANGE, height=0.55, label="Needs review",
            edgecolor=SURFACE, linewidth=2)
    for yi, a, b in zip(y, conf, rev):
        ax.text(a + b + 0.3, yi, f"{a}" + (f" (+{b})" if b else ""), va="center", color=INK2, fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels([SHORT[t] for t in types], color=INK)
    ax.set_xlabel("Flagged elements", color=INK2, fontsize=10)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, labelcolor=INK2,
              handlelength=1.2, borderaxespad=0.3)
    ax.set_xlim(0, max([a + b for a, b in zip(conf, rev)] + [1]) * 1.22)
    ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_heatmap(A, path):
    types = [p for p in ORDER if any(A["matrix"][s][p] or A["matrix_review"][s][p] for s in SECTORS)] or ORDER[:3]
    data = [[A["matrix"][s][p] / (A["sector_n"][s] or 1) for p in types] for s in SECTORS]
    fig, ax = plt.subplots(figsize=(1.6 * len(types) + 1.8, 2.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(data, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(SECTORS):
        for j, p in enumerate(types):
            k, n, r = A["matrix"][s][p], A["sector_n"][s], A["matrix_review"][s][p]
            ax.text(j, i, f"{k}/{n}" + (f"  (+{r})" if r else ""), ha="center", va="center", fontsize=10,
                    color="#ffffff" if data[i][j] >= 0.55 else INK)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([SHORT[t].replace(" ", "\n", 1) for t in types], color=INK, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels(SECTORS, color=INK, fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(types))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(SECTORS))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- html pieces
def pct(k, n, *_):
    return f"{k}/{n} ({round(100 * k / n) if n else 0}%)"


def table_sites(A):
    types = [p for p in ORDER if any(A["per_site"][s][p] or A["per_site_review"][s][p] for s in A["sites"])]
    head = "".join(f"<th>{SHORT[t]}</th>" for t in types)
    rows = []
    for sec in SECTORS:
        for s in [x for x in A["sites"] if A["sector_of"][x] == sec]:
            cells = ""
            for t in types:
                a, b = A["per_site"][s][t], A["per_site_review"][s][t]
                txt = (str(a) if a else "·") + (f'<span class="rev"> +{b}</span>' if b else "")
                cells += f'<td class="num{" hit" if a else ""}">{txt}</td>'
            total = sum(A["per_site"][s].values())
            rows.append(f"<tr><td>{e(s)}</td><td class='muted'>{sec}</td>{cells}"
                        f"<td class='num'><b>{total}</b></td><td class='num'>{len(A['per_site'][s])}</td></tr>")
    return (f"<table><thead><tr><th>Site</th><th>Sector</th>{head}<th>Total</th><th>Types</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def table_method(metas):
    rows = []
    for m in metas:
        pages = ", ".join(p["page_type"] for p in m["pages"]) or "—"
        skipped = [l for l in m["log"] if l["status"].startswith(("skipped", "error", "http"))]
        note = "; ".join(sorted({l["status"] for l in skipped})) or ""
        rows.append(f"<tr><td>{e(m['site'])}</td><td class='muted'>{m['sector']}</td>"
                    f"<td class='num'>{len(m['pages'])}</td><td>{e(pages)}</td>"
                    f"<td class='muted'>{e(m['started'][:16].replace('T', ' '))}</td><td class='muted small'>{e(note)}</td></tr>")
    return ("<table><thead><tr><th>Site</th><th>Sector</th><th>Pages</th><th>Page types</th>"
            "<th>Scraped (IST)</th><th>Notes</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def table_validation(V):
    if not V:
        return ("<p class='callout'><b>Pending.</b> Run <code>python run.py validate-sample</code>, label the 30 rows in "
                "<code>data/validation_sample.csv</code> by hand, then regenerate this report.</p>")
    if not V["labelled"]:
        return (f"<p class='callout'><b>Pending.</b> A {V['sample_size']}-element sample has been drawn "
                "(<code>data/validation_sample.csv</code>) but not yet labelled by hand. Fill in "
                "<code>human_is_dark</code> (1/0) and regenerate this report.</p>")
    f = lambda v: "—" if v is None else f"{v:.2f}"
    return f"""<table class="narrow"><tbody>
<tr><td>Elements hand-labelled</td><td class="num">{V['labelled']} of {V['sample_size']}</td></tr>
<tr><td>True positives / false positives</td><td class="num">{V['tp']} / {V['fp']}</td></tr>
<tr><td>False negatives / true negatives</td><td class="num">{V['fn']} / {V['tn']}</td></tr>
<tr><td><b>Precision</b></td><td class="num"><b>{f(V['precision'])}</b></td></tr>
<tr><td>Recall (within sample)</td><td class="num">{f(V['recall_within_sample'])}</td></tr>
<tr><td>Accuracy</td><td class="num">{f(V['accuracy'])}</td></tr>
<tr><td>Pattern-type agreement (n={V['pattern_type_n']})</td><td class="num">{f(V['pattern_type_agreement'])}</td></tr>
</tbody></table>"""


def gallery(cands, k=6):
    pool = [c for c in cands if counts_dp(c) and c.get("crop_path")]
    if len(pool) < k:  # top up with the strongest low-confidence flags, clearly marked
        pool += [c for c in cands if review_dp(c) and c.get("crop_path")]
    pool.sort(key=lambda c: (not counts_dp(c), -c["label"]["confidence"]))
    chosen, seen_t, seen_s = [], Counter(), Counter()
    for rnd in range(3):  # prefer variety: new pattern type, then new site
        for c in pool:
            if len(chosen) >= k or c in chosen:
                continue
            t, s = c["label"]["pattern_type"], c["site"]
            if (rnd == 0 and (seen_t[t] or seen_s[s])) or (rnd == 1 and seen_t[t] >= 2):
                continue
            chosen.append(c)
            seen_t[t] += 1
            seen_s[s] += 1
    if not chosen:
        return "<p class='muted'>No high-confidence flagged elements with a screenshot crop in this run.</p>"
    cards = []
    for c in chosen:
        L = c["label"]
        src = f"crops/{c['id']}.jpg"
        shutil.copy(config.ROOT / c["crop_path"], config.REPORT_DIR / src)
        cards.append(f"""<figure class="ev">
  <div class="shot"><img src="{src}" alt="{e(c['text'][:80])}"></div>
  <figcaption>
    <div class="tags"><span class="tag">CCPA: {e(L['ccpa_label'])}</span><span class="tag alt">OECD: {e(L['oecd_category'])}</span>{'<span class="tag warn">needs review</span>' if L['needs_review'] else ''}</div>
    <p class="q">“{e(c['text'][:140])}”</p>
    <p>{e(L['reason'])}</p>
    <p class="muted small">{e(c['site'])} · {c['page_type']} page · confidence {L['confidence']:.2f} · {e(c['timestamp'][:16].replace('T', ' '))}</p>
  </figcaption></figure>""")
    return f"<div class='gallery'>{''.join(cards)}</div>"


CSS = """
:root{--ink:#111a24;--ink2:#4b5867;--muted:#7c8896;--grid:#e2e7ee;--surface:#ffffff;--plane:#f4f6f9;--soft:#eef2f7;
  --blue:#2a78d6;--blue-soft:#e6f0fc;--orange:#eb6834;--orange-soft:#fdeee6;color-scheme:light}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--plane);color:var(--ink);font:15.5px/1.65 "Source Sans 3",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1000px;margin:0 auto;padding-block:56px 80px}
h1,h2,h3{font-family:"Bricolage Grotesque","Avenir Next","Segoe UI",sans-serif;text-wrap:balance}
.eyebrow{font:700 12px/1 "Source Sans 3",sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--blue);margin-bottom:12px}
h1{font-size:clamp(30px,4.5vw,44px);line-height:1.08;margin:0 0 8px;letter-spacing:-.02em}
h2{font-size:24px;margin:56px 0 14px;letter-spacing:-.01em;scroll-margin-top:16px}
h3{font-size:16.5px;margin:32px 0 10px}
p{max-width:78ch}
.sub{color:var(--ink2);margin:0 0 20px}
nav.sec{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px}
nav.sec a{font-size:13.5px;font-weight:600;color:var(--ink2);text-decoration:none;background:var(--surface);border:1px solid var(--grid);padding:6px 12px;border-radius:99px}
nav.sec a:hover{border-color:var(--blue);color:var(--blue)}
.muted{color:var(--muted)} .small{font-size:12.5px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 20px}
.kpi{background:var(--surface);border:1px solid var(--grid);border-radius:14px;padding:16px 18px;display:grid;gap:4px;
  box-shadow:0 1px 2px rgba(17,26,36,.04),0 8px 24px -14px rgba(17,26,36,.14)}
.kpi .k{font-size:11.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.kpi b{font:700 30px/1.1 "Bricolage Grotesque",sans-serif;font-variant-numeric:tabular-nums}
.kpi span{color:var(--ink2);font-size:13.5px;line-height:1.4}
.kpi em{font-style:normal;color:var(--orange);font-weight:600}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:separate;border-spacing:0;background:var(--surface);border:1px solid var(--grid);border-radius:12px;overflow:hidden;font-size:13.5px;margin:8px 0}
table.narrow{max-width:600px}
th,td{padding:9px 12px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:#fafbfd}
th{font-weight:700;color:var(--ink2);font-size:12px;letter-spacing:.02em;background:var(--soft)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.hit{color:var(--blue);font-weight:700}
.rev{color:var(--orange);font-weight:600;font-size:12px}
img.fig{width:100%;border:1px solid var(--grid);border-radius:12px;background:#fcfcfb;padding:8px}
.caption{color:var(--ink2);font-size:13px;margin:8px 0 0}
.callout{background:#fff7ec;border:1px solid #f3d9a8;border-radius:12px;padding:12px 16px;max-width:none}
.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.ev{margin:0;background:var(--surface);border:1px solid var(--grid);border-radius:14px;overflow:hidden;display:grid;grid-template-rows:auto 1fr}
.ev .shot{background:#f1f3f6;display:flex;justify-content:center;align-items:center;height:220px;overflow:hidden;border-bottom:1px solid var(--grid)}
.ev img{max-width:100%;max-height:100%;object-fit:contain}
.ev figcaption{padding:14px 16px}
.ev p{margin:6px 0}
.q{font-weight:700}
.tags{display:flex;flex-wrap:wrap;gap:6px}
.tag{display:inline-block;font-size:11.5px;font-weight:600;padding:3px 9px;border-radius:99px;background:var(--blue-soft);color:#184f95}
.tag.alt{background:var(--soft);color:var(--ink2)}
.tag.warn{background:var(--orange-soft);color:#a3401a}
code{background:var(--soft);padding:1px 6px;border-radius:5px;font-size:12.5px}
ul{padding-left:20px} ul li{margin:6px 0;max-width:78ch}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.gallery{grid-template-columns:1fr}}
"""


def run():
    cands, metas = load()
    A = analyze(cands, metas)
    V = validate.scores()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "crops").mkdir(exist_ok=True)
    fig_frequency(A, config.REPORT_DIR / "fig1_frequency.png")
    fig_heatmap(A, config.REPORT_DIR / "fig2_heatmap.png")

    n_pages = sum(len(m["pages"]) for m in metas)
    stamps = sorted(p["timestamp"] for m in metas for p in m["pages"])
    window = f"{stamps[0][:16].replace('T', ' ')} – {stamps[-1][11:16]}" if stamps else "—"
    backend = cands[0]["label"]["backend"] if cands else "—"
    checks_file = config.DATA / "countdown_checks.json"
    checks = json.loads(checks_file.read_text()) if checks_file.exists() else {}
    n_review = sum(review_dp(c) for c in cands)
    n_count = sum(counts_dp(c) for c in cands)
    n_sites = len(A["sites"])
    prev_rows = "".join(f"<tr><td>{k}</td><td class='num'>{pct(v[0], v[1])}</td><td class='num muted'>{pct(v[2], v[1])}</td></tr>"
                        for k, v in A["prevalence"].items())
    ov = A["prevalence"]["Overall"]
    share = lambda a, b: f"{round(100 * a / b)}%" if b else "—"
    backend_note = ("" if not backend.startswith("keyword") else
        "<p class='callout'><b>Classifier: keyword baseline.</b> No LLM key was found, so a transparent rule set labelled the "
        "elements. It cannot judge context, so most flags sit below the confidence threshold and are shown as <i>needs review</i>. "
        "Set <code>NEBIUS_API_KEY</code> and <code>NEBIUS_MODEL</code> (in <code>.env.local</code>) and re-run "
        "<code>python run.py classify</code> then <code>python run.py report</code>.</p>")
    top = A["freq_sites"].most_common(3)
    top_txt = ", ".join(f"{SHORT[p].lower()} ({k} of {n_sites} sites)" for p, k in top) or "none"
    cd_rows = "".join(
        f"<tr><td>{e(s)}</td><td class='num'>{v['observations']}</td><td class='num'>{v['spread_s'] if v['spread_s'] is not None else '—'}</td><td>{e(v['verdict'])}</td></tr>"
        for s, v in checks.items())
    cd_table = (f"<h3>Countdown consistency check</h3><p>For each countdown seen, implied end time = snapshot time + time remaining. "
                f"If a timer counts toward a real deadline, these agree across page loads.</p>"
                f"<table class='narrow'><thead><tr><th>Site</th><th class='num'>Timers seen</th><th class='num'>End-time spread (s)</th><th>Verdict</th></tr></thead>"
                f"<tbody>{cd_rows}</tbody></table>") if cd_rows else ""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dark Patterns in Indian D2C E-commerce — Pilot Findings</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700&family=Source+Sans+3:wght@400;600;700&display=swap"><style>{CSS}</style></head><body><main>
<div class="eyebrow">Pilot findings · automated detection</div>
<h1>Dark Patterns in Indian D2C E-commerce</h1>
<p class="sub">Pilot run of the automated detection pipeline · generated {datetime.now().strftime('%d %b %Y, %H:%M')}</p>

<nav class="sec"><a href="#method">Sample &amp; method</a><a href="#findings">Findings</a><a href="#evidence">Evidence</a><a href="#limits">Limitations</a><a href="#next">Next steps</a></nav>
<div class="kpis">
  <div class="kpi"><span class="k">Sample</span><b>{n_sites} sites</b><span>{n_pages} pages · 3 sectors</span></div>
  <div class="kpi"><span class="k">Prevalence</span><b>{share(ov[0], ov[1])}</b><span>of sites with ≥1 counted pattern<br><em>{share(ov[2], ov[1])} incl. needs review</em></span></div>
  <div class="kpi"><span class="k">Flagged elements</span><b>{n_count}</b><span>counted<br><em>+{n_review} needs review</em></span></div>
  <div class="kpi"><span class="k">Candidates</span><b>{len(cands)}</b><span>extracted by rules<br><em>classifier: {e(backend)}</em></span></div>
</div>
{backend_note}

<h2 id="method">1. Sample and method</h2>
<p><b>Sample.</b> {n_sites} Indian D2C / e-commerce sites across three sectors (Electronics, Fashion, Cosmetics). For each site
the pipeline visited the homepage, one category page, two product pages and, where robots.txt allows it, the cart page after
adding one item. Scrape window: {window} IST.</p>
<p><b>Pipeline.</b> Collect (Playwright headless Chromium renders JavaScript; HTML, full-page screenshot, URL and timestamp saved)
→ Extract (BeautifulSoup rules for urgency/scarcity, countdowns, social proof, strikethrough discounts, fee lines,
pre-ticked add-ons, confirm-shaming wording, subscription terms) → Store (ChromaDB, one embedded record per candidate)
→ Classify (<code>{e(backend)}</code> against CCPA 2023 and OECD definitions; confidence &lt; {config.CONFIDENCE_THRESHOLD} = needs review)
→ Analyze (descriptive statistics).</p>
<p><b>Ethics.</b> Public pages only; no logins, no personal data entered, no orders placed. robots.txt was checked before every request
and disallowed paths were skipped; 2–3 s pause between requests. Every page is timestamped because sites change.</p>
<div class="scroll">{table_method(metas)}</div>

<h2 id="findings">2. Findings</h2>
<p>Counting rule: an element counts when the classifier labels it a dark pattern with confidence ≥ {config.CONFIDENCE_THRESHOLD}.
Lower-confidence flags are reported separately (<span class="rev">orange / +n</span>) and are excluded from headline figures.
The most widespread patterns were {top_txt}.</p>

<h3>Prevalence: sites with at least one dark pattern</h3>
<table class="narrow"><thead><tr><th>Group</th><th class="num">Counted</th><th class="num">Incl. needs review</th></tr></thead><tbody>{prev_rows}
<tr><td>Mean flagged elements per site</td><td class="num">{A['avg_elements_per_site']}</td><td></td></tr>
<tr><td>Mean distinct pattern types per site</td><td class="num">{A['avg_types_per_site']}</td><td></td></tr></tbody></table>

<h3>Table 1. Patterns detected per site (flagged elements)</h3>
<div class="scroll">{table_sites(A)}</div>
<p class="caption">“·” = none counted. <span class="rev">+n</span> = additional low-confidence flags awaiting review.</p>

<h3>Figure 1. Pattern frequency</h3>
<img class="fig" src="fig1_frequency.png" alt="Bar chart of flagged elements by pattern type">
<p class="caption">Element counts are inflated by repetition (the same badge on many product tiles); Figure 2 counts sites instead.</p>

<h3>Figure 2. Sector × pattern</h3>
<img class="fig" src="fig2_heatmap.png" alt="Heatmap of sites per sector showing each pattern">
<p class="caption">Cell = sites in that sector with ≥1 counted instance; (+n) = further sites with only needs-review flags. With 3–4 sites per sector these are descriptive only; no significance tests.</p>
{cd_table}

<h3>Table 2. Classifier validation</h3>
{table_validation(V)}

<h2 id="evidence">3. Evidence gallery</h2>
{gallery(cands)}

<h2 id="limits">4. Limitations</h2>
<ul>
<li><b>Small sample.</b> {n_sites} sites and ≤5 pages each; proportions can swing by 25 points with one site.</li>
<li><b>Rule-based extraction misses patterns.</b> Only elements matching the extraction rules reach the classifier; images-of-text,
novel wording and patterns spread across several elements are missed, so prevalence is a lower bound.</li>
<li><b>Hidden charges after checkout starts.</b> Fees that appear only after entering an address or payment details cannot be
observed without placing an order, which this study does not do. Several sites also disallow <code>/cart</code> in robots.txt,
so drip pricing is under-observed.</li>
<li><b>Snapshot in time.</b> Sale banners, timers and stock badges change hourly; results describe the scrape window only.</li>
<li><b>Truth of claims is unverifiable.</b> A scraper cannot know whether “only 3 left” is true; such flags carry lower confidence.</li>
<li><b>The classifier can mislabel.</b> See Table 2; findings should be read alongside the hand-validated precision.</li>
</ul>

<h2 id="next">5. Next steps</h2>
<ul>
<li>Scale to 50 sites with the same pipeline; add repeated scrapes over several days to measure timer resets and stock-count changes.</li>
<li>Run inferential tests across sectors (e.g. chi-square / Fisher's exact) once cell counts allow.</li>
<li>Run the two-group experiment (manipulative vs. honest interface) to measure the effect on purchase decisions.</li>
</ul>
<p class="muted small">Sources: Central Consumer Protection Authority, Guidelines for Prevention and Regulation of Dark Patterns, 2023;
OECD (2022), Dark Commercial Patterns, OECD Digital Economy Papers No. 336.</p>
</main></body></html>"""
    out = config.REPORT_DIR / "report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[report] wrote {out.relative_to(config.ROOT)}")
    return out
