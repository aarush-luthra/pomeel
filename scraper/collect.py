"""Stage 1: Collect.

Loads public pages in a real (headless) Chromium so JavaScript-rendered timers
and stock badges appear, then saves for every page:
  - the full rendered HTML (annotated with element bounding boxes)
  - a full-page screenshot
  - meta.json: url, sector, page type, timestamp, HTTP status

Ethics: checks robots.txt before every request, skips disallowed paths,
waits 2-3 s between requests, never logs in, never enters personal data,
never goes past the cart page.
"""
import json
import random
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

import config

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Runs inside the page after it settles. Writes each visible element's
# document-coordinate box into data-dp-bbox so the parser (BeautifulSoup) can
# later crop the screenshot, and records computed state that HTML alone loses:
# strikethrough styling and the live "checked" property of checkboxes.
ANNOTATE_JS = r"""
() => {
  const sx = window.scrollX, sy = window.scrollY;
  let n = 0;
  for (const el of document.body.querySelectorAll('*')) {
    const tag = el.tagName;
    if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' || tag === 'SVG') continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || cs.opacity === '0') continue;
    el.setAttribute('data-dp-bbox',
      [r.left + sx, r.top + sy, r.width, r.height].some(v => !isFinite(v)) ? '' : [Math.round(r.left + sx), Math.round(r.top + sy), Math.round(r.width), Math.round(r.height)].join(','));
    if ((cs.textDecorationLine || cs.textDecoration || '').includes('line-through'))
      el.setAttribute('data-dp-strike', '1');
    if (tag === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio'))
      el.setAttribute('data-dp-checked', el.checked ? '1' : '0');
    n++;
  }
  return n;
}
"""

ADD_TO_CART = re.compile(r"^\s*(add\s+to\s+(cart|bag|basket)|buy\s+now)\s*$", re.I)


def polite_wait():
    time.sleep(random.uniform(config.MIN_DELAY_S, config.MAX_DELAY_S))


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Robots:
    """robots.txt check, fetched through the browser context (same UA as pages)."""

    def __init__(self, ctx, base_url):
        self.rp = RobotFileParser()
        robots_url = urljoin(base_url, "/robots.txt")
        self.status = None
        try:
            resp = ctx.request.get(robots_url, timeout=20_000)
            self.status = resp.status
            if resp.status in (401, 403):
                self.rp.disallow_all = True
            elif resp.ok:
                self.rp.parse(resp.text().splitlines())
            else:
                self.rp.allow_all = True   # no robots.txt -> everything allowed
        except Exception:
            self.rp.allow_all = True
            self.status = "error"

    def allowed(self, url):
        return self.rp.can_fetch("*", url)


def settle_and_scroll(page):
    """Wait for JS, scroll through the page so lazy content renders, return to top."""
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except PWTimeout:
        pass
    height = page.evaluate("document.body ? document.body.scrollHeight : 0") or 0
    y = 0
    while y < min(height, 15_000):
        y += 900
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(250)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(1500)


def snapshot(page, site, page_type, out_dir, idx):
    """Annotate DOM, save HTML + full-page screenshot + meta. Returns meta dict."""
    settle_and_scroll(page)
    page.evaluate(ANNOTATE_JS)
    stem = f"{idx:02d}_{page_type}"
    html_path = out_dir / f"{stem}.html"
    shot_path = out_dir / f"{stem}.png"
    html_path.write_text(page.content(), encoding="utf-8")
    try:
        page.screenshot(path=str(shot_path), full_page=True, timeout=30_000)
    except Exception:
        page.screenshot(path=str(shot_path), timeout=30_000)
    return {
        "site": site["name"],
        "sector": site["sector"],
        "page_type": page_type,
        "url": page.url,
        "title": page.title(),
        "timestamp": now_iso(),
        "html_path": str(html_path.relative_to(config.ROOT)),
        "screenshot_path": str(shot_path.relative_to(config.ROOT)),
    }


def find_links(page, base_url, hints, limit, exclude=()):
    host = urlparse(base_url).netloc.replace("www.", "")
    hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    seen, out = set(exclude), []
    for h in hrefs:
        h = h.split("#")[0].split("?")[0]
        if host not in urlparse(h).netloc:
            continue
        if "gift-card" in h or "giftcard" in h:
            continue
        if any(x in h for x in hints) and h not in seen:
            seen.add(h)
            out.append(h)
        if len(out) >= limit:
            break
    return out


def visit(page, url, robots, log):
    if not robots.allowed(url):
        log.append({"url": url, "status": "skipped: disallowed by robots.txt"})
        return False
    polite_wait()
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
    except Exception as e:
        log.append({"url": url, "status": f"error: {type(e).__name__}"})
        return False
    code = resp.status if resp else None
    if code and code >= 400:
        log.append({"url": url, "status": f"http {code}"})
        return False
    log.append({"url": url, "status": f"ok ({code})"})
    return True


def try_add_to_cart(page):
    for el in page.query_selector_all("button, input[type=submit], a"):
        try:
            txt = (el.inner_text() if el.evaluate("e => e.tagName") != "INPUT"
                   else el.get_attribute("value")) or ""
            if ADD_TO_CART.match(txt) and el.is_visible():
                el.click(timeout=5_000)
                page.wait_for_timeout(3000)
                return True
        except Exception:
            continue
    return False


def collect_site(browser, site):
    out_dir = config.RAW / config.slug(site["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900},
                              locale="en-IN", timezone_id="Asia/Kolkata")
    page = ctx.new_page()
    log, pages = [], []
    robots = Robots(ctx, site["url"])
    result = {"site": site["name"], "sector": site["sector"], "robots_status": robots.status,
              "started": now_iso(), "pages": pages, "log": log}
    try:
        # 1. homepage
        if not visit(page, site["url"], robots, log):
            return result
        pages.append(snapshot(page, site, "home", out_dir, 1))
        home_url = page.url

        # 2. category page
        cats = find_links(page, home_url, site["category_hint"], 1)
        products = []
        if cats and visit(page, cats[0], robots, log):
            pages.append(snapshot(page, site, "category", out_dir, 2))
            products = find_links(page, home_url, site["product_hint"], 2)
        if not products:  # fall back to products linked from wherever we are / homepage
            if page.url != home_url:
                visit(page, home_url, robots, log)
            products = find_links(page, home_url, site["product_hint"], 2)

        # 3. two product pages
        for i, purl in enumerate(products[:2]):
            if visit(page, purl, robots, log):
                pages.append(snapshot(page, site, f"product{i + 1}", out_dir, 3 + i))

        # 4. cart page — only if robots.txt allows the cart path
        cart_url = urljoin(home_url, site.get("cart_path", "/cart"))
        if not robots.allowed(cart_url):
            log.append({"url": cart_url, "status": "skipped: cart disallowed by robots.txt"})
        elif products and visit(page, products[0], robots, log):
            added = try_add_to_cart(page)
            log.append({"url": products[0], "status": f"add-to-cart {'clicked' if added else 'button not found'}"})
            if visit(page, cart_url, robots, log):
                pages.append(snapshot(page, site, "cart", out_dir, 5))
    except Exception as e:
        log.append({"url": page.url, "status": f"error: {type(e).__name__}: {str(e)[:120]}"})
    finally:
        result["finished"] = now_iso()
        (out_dir / "meta.json").write_text(json.dumps(result, indent=2))
        ctx.close()
    return result


def run(only=None, headless=True):
    config.RAW.mkdir(parents=True, exist_ok=True)
    sites = config.load_sites()
    if only:
        sites = [s for s in sites if s["name"].lower() in {o.lower() for o in only}]
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        for site in sites:
            print(f"[collect] {site['name']} ({site['sector']})", flush=True)
            r = collect_site(browser, site)
            print(f"          {len(r['pages'])} pages saved; robots.txt={r['robots_status']}", flush=True)
            for entry in r["log"]:
                print(f"          - {entry['status']}: {entry['url']}", flush=True)
            results.append(r)
        browser.close()
    return results
