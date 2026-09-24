import os
from datetime import timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# load .env file if present
_env_file = ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def env_str(key: str, default: str = "") -> str:
    """Read env var, falling back to default when unset OR empty."""
    return os.environ.get(key, "").strip() or default


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# User workflow timezone: India Standard Time.
INDIA_TZ = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
# Backwards-compatible alias used by the original project modules.
BEIJING_TZ = INDIA_TZ

# --- LLM ---
# Any OpenAI-compatible provider. LLM_* names are preferred; DEEPSEEK_* work
# as legacy aliases.
LLM_API_KEY = env_str("LLM_API_KEY") or env_str("DEEPSEEK_API_KEY")
LLM_BASE_URL = env_str("LLM_BASE_URL") or env_str("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = env_str("LLM_MODEL", "openrouter/free")

# --- Content Sources ---
SOURCES = {
    "hn_top": "https://hacker-news.firebaseio.com/v0/topstories.json",
    "devto": "https://dev.to/api/articles?top=10",
    "arxiv_cs_ai": "https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=lastUpdatedDate&max_results=5",
}

RSS_FEEDS = [
    "https://hnrss.org/frontpage?count=10",
    "https://techcrunch.com/feed/",
    "https://simonwillison.net/atom/everything/",
    "https://lobste.rs/rss",
]

MAX_ARTICLES_PER_SOURCE = 10
MAX_TOTAL_ARTICLES = 20

# --- Publishing (optional) ---
TELEGRAM_BOT_TOKEN = env_str("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL = env_str("TELEGRAM_CHANNEL")

WP_URL = env_str("WP_URL")
WP_USER = env_str("WP_USER")
WP_APP_PASSWORD = env_str("WP_APP_PASSWORD")

FEISHU_WEBHOOK_URL = env_str("FEISHU_WEBHOOK_URL")
FEISHU_SECRET = env_str("FEISHU_SECRET")

# --- Runtime ---
SEEN_URLS_FILE = DATA_DIR / "seen_urls.json"
RAW_ARTICLES_FILE = DATA_DIR / "raw_articles.json"
DIGEST_OUTPUT = OUTPUT_DIR / "digest.md"
SOCIAL_OUTPUT = OUTPUT_DIR / "social_posts.json"
INSTAGRAM_PACK_OUTPUT = OUTPUT_DIR / "instagram_daily_pack.md"
INSTAGRAM_PACK_JSON = OUTPUT_DIR / "instagram_daily_pack.json"


def validate_config() -> None:
    """Fail fast with a clear message when required config is missing."""
    missing = []
    if not LLM_API_KEY:
        missing.append("LLM_API_KEY (or legacy DEEPSEEK_API_KEY)")
    if missing:
        raise SystemExit(
            "[config] missing required env vars: "
            + ", ".join(missing)
            + " — set them in .env (local) or repo secrets (CI)"
        )