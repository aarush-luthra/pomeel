"""Shared paths and settings for the dark-pattern pipeline."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # one folder per site: html, screenshots, meta
CROPS = DATA / "crops"        # screenshot crops of candidate elements
CHROMA_DIR = DATA / "chroma"
CANDIDATES = DATA / "candidates.jsonl"
CLASSIFIED = DATA / "classified.jsonl"
VALIDATION_CSV = DATA / "validation_sample.csv"
REPORT_DIR = ROOT / "report"

SITES_FILE = ROOT / "sites.json"

# Politeness
MIN_DELAY_S = 2.0
MAX_DELAY_S = 3.0
PAGE_TIMEOUT_MS = 45_000



def _load_dotenv():
    """Read KEY=VALUE lines from .env.local / .env (scraper/ or project root); real env vars win."""
    for f in (ROOT / ".env.local", ROOT.parent / ".env.local", ROOT / ".env", ROOT.parent / ".env"):
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v and not os.environ.get(k.strip()):
                        os.environ[k.strip()] = v


_load_dotenv()

# Classifier: Nebius AI Studio (OpenAI-compatible API) is the primary LLM backend
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")
NEBIUS_MODEL = os.environ.get("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
OPENAI_MODEL = os.environ.get("DP_OPENAI_MODEL", "gpt-4o-mini")
CONFIDENCE_THRESHOLD = 0.7    # below this -> "needs review"


def load_sites():
    return json.loads(SITES_FILE.read_text())["sites"]


def slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
