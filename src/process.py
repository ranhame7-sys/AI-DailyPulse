import json
from datetime import datetime

from openai import OpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    BEIJING_TZ,
    RAW_ARTICLES_FILE,
    DIGEST_OUTPUT,
    SOCIAL_OUTPUT,
    INSTAGRAM_PACK_OUTPUT,
    INSTAGRAM_PACK_JSON,
)

_client = None


def _get_client() -> OpenAI:
    """Lazy client: created on first LLM call, so --collect-only works without a key."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


def _chat(system: str, user: str, temperature: float = 0.3, max_tokens: int = 2000, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content


def _parse_json(text: str) -> dict:
    """Parse a JSON object out of an LLM reply (with fence fallback)."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


# ─── Step 1: Filter + Summarize (one LLM call per article) ──

DIGEST_PROMPT = """You are an AI-news editor for an Instagram page focused on useful, current AI tools, product launches, model releases, creator workflows, and major AI updates.

Decide whether the article is worth including in today's creator-focused AI content pack.

Reject when:
- it is pure marketing, sponsored copy, or low-information hype
- it is unrelated to AI, software, creator tools, or meaningful tech developments
- it repeats an already-covered story without new information
- it is mainly opinion with no concrete update

Keep when:
- a new AI model, product, feature, tool, workflow, benchmark, or important policy/update is announced
- the story gives creators or everyday AI users something practical to understand or try
- it is a significant industry development with clear user impact

For kept stories output:
- title_cn: concise English headline, max 90 characters (field name kept for compatibility)
- key_points: max 2 factual points, each concise
- one_liner: one-sentence explanation of why it matters
- relevance_score: 0.0 to 1.0, where 1.0 is highly useful for an AI-focused Instagram audience

Rules:
- Do not invent facts.
- Preserve product/model/company names exactly.
- Keep numbers and dates accurate.
- Prefer concrete usefulness over hype.

Output JSON only:
{"relevant": true, "reason": "short reason", "title_cn": "English headline", "key_points": ["point 1", "point 2"], "one_liner": "why it matters", "relevance_score": 0.8}"""


def _fallback_digest(article: dict) -> dict:
    return {
        "title_cn": article["title"],
        "key_points": [article.get("summary", "")[:60]],
        "one_liner": article["title"],
        "relevance_score": 0.5,
    }


def process_articles(articles: list[dict]) -> tuple[list[dict], int]:
    """Filter + summarize each article in a single LLM call, drop irrelevant ones.

    Returns (kept_articles, error_count). A failed call keeps the article with a
    fallback digest; the caller decides whether the error ratio is acceptable.
    """
    kept = []
    errors = 0
    for i, a in enumerate(articles):
        print(f"[process] digesting {i+1}/{len(articles)}: {a['title'][:50]}")
        text = f"标题: {a['title']}\n来源: {a['source']}\n摘要: {a.get('summary', '')[:1500]}"
        try:
            data = _parse_json(_chat(DIGEST_PROMPT, text, temperature=0.1, max_tokens=600, json_mode=True))
            if not data.get("relevant"):
                continue
            a["filter_reason"] = data.get("reason", "")
            points = [str(p) for p in (data.get("key_points") or []) if str(p).strip()]
            a["digest"] = {
                "title_cn": str(data.get("title_cn") or a["title"])[:50],
                "key_points": points[:2],
                "one_liner": str(data.get("one_liner") or "")[:60],
                "relevance_score": float(data.get("relevance_score") or 0.5),
            }
        except Exception as e:
            print(f"[process] digest error for '{a['title'][:40]}': {e}")
            # LLM failure shouldn't drop the article
            errors += 1
            a["digest"] = _fallback_digest(a)
        kept.append(a)
    print(f"[process] filtered: {len(articles)} → {len(kept)} ({errors} errors)")
    return kept, errors


# abort publishing when this fraction of LLM calls fails — a full API outage
# shouldn't push a digest full of fallback placeholders to channels
MAX_ERROR_RATIO = 0.5


# ─── Step 2: Finalize (one editor call → headline + 4 picks) ──

FINALIZE_PROMPT = """You are the editor of a daily AI Instagram page.

From the candidate stories below, choose exactly 5 of the strongest stories for today's content pack.

Selection priorities:
1. New AI tools, major feature launches, model releases, creator workflows, or important updates.
2. Practical usefulness to creators and everyday AI users.
3. Freshness and significance.
4. Avoid five stories that are all about the same company or topic when possible.

Return:
- headline: the strongest story, with:
  - headline_title: clear English headline
  - headline_paragraph: 2-3 factual sentences explaining the update and why it matters
  - url: must exactly match a candidate URL
- items: exactly 4 additional stories, each with:
  - title_cn: concise English title (field name kept for compatibility)
  - blurb: 1-2 factual sentences
  - url: must exactly match a candidate URL

Do not invent links, dates, capabilities, prices, benchmarks, or quotes.
Prefer official/company/news sources over social posts where possible.

Output JSON only:
{"headline": {"url": "...", "headline_title": "...", "headline_paragraph": "..."},
 "items": [{"url": "...", "title_cn": "...", "blurb": "..."}]}"""


def _candidate_section(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        d = a.get("digest", {})
        lines.append(
            f"[{i}] 标题: {a['title']} | 中文: {d.get('title_cn', '')} | "
            f"要点: {'；'.join(d.get('key_points', []))} | 一句话: {d.get('one_liner', '')} | "
            f"来源: {a['source']} | 链接: {a['url']}"
        )
    return "\n".join(lines)


def finalize(articles: list[dict]) -> dict:
    """Pick the headline story + 4 picks from the candidates; sanitize the model output."""
    by_url = {a["url"]: a for a in articles}
    ranked = sorted(articles, key=lambda a: a.get("digest", {}).get("relevance_score", 0), reverse=True)

    def _fallback() -> dict:
        top = ranked[0]
        d = top["digest"]
        return {
            "headline": {
                "url": top["url"],
                "headline_title": d.get("title_cn", top["title"]),
                "headline_paragraph": f"{d.get('one_liner', '')}。{d.get('key_points', [''])[0]}",
            },
            "items": [],
        }

    try:
        data = _parse_json(_chat(
            FINALIZE_PROMPT, _candidate_section(articles),
            temperature=0.2, max_tokens=1200, json_mode=True,
        ))
    except Exception as e:
        print(f"[process] finalize error, using fallback picks: {e}")
        return _fallback()

    headline = dict(data.get("headline") or {})
    ha = by_url.get(str(headline.get("url", "")))
    if not ha:  # model invented a url — take the top-ranked candidate
        ha = ranked[0]
        headline["url"] = ha["url"]
    hd = ha["digest"]
    if not headline.get("headline_title"):
        headline["headline_title"] = hd.get("title_cn", ha["title"])
    if not headline.get("headline_paragraph"):
        headline["headline_paragraph"] = f"{hd.get('one_liner', '')}。{hd.get('key_points', [''])[0]}"

    items, seen_urls = [], {headline["url"]}
    for it in data.get("items") or []:
        url = str(it.get("url", ""))
        if url in seen_urls or url not in by_url:
            continue
        seen_urls.add(url)
        items.append({
            "url": url,
            "title_cn": str(it.get("title_cn") or by_url[url]["digest"].get("title_cn", ""))[:50],
            "blurb": str(it.get("blurb") or by_url[url]["digest"].get("one_liner", ""))[:80],
        })
        if len(items) == 4:
            break
    # fill missing picks from the ranking (prefer error-free LLM output but never starve the list)
    for a in ranked:
        if len(items) >= 4:
            break
        if a["url"] in seen_urls:
            continue
        seen_urls.add(a["url"])
        items.append({
            "url": a["url"],
            "title_cn": a["digest"].get("title_cn", a["title"])[:50],
            "blurb": a["digest"].get("one_liner", "")[:80],
        })

    return {"headline": headline, "items": items}


# ─── Step 3: Headline Rewrite (social posts) ────────────

STYLE_PROMPTS = {
    "instagram_caption": """Write an Instagram caption for an AI-news post.
Start with a strong but truthful hook, then explain the update in simple English.
Keep it concise and useful. Include one natural CTA at the end.
Add 5-8 relevant hashtags. Do not exaggerate or invent facts.""",

    "reel_script": """Write a short Instagram Reel script about this AI update.
Use: hook -> what happened -> why it matters -> who should care -> simple CTA.
Natural spoken English, roughly 30-45 seconds. No editing directions. No hype that is not supported by the source.""",
}


def generate_headline_posts(headline: dict, article: dict) -> dict:
    """Generate multi-platform social posts for the headline story."""
    content = f"""头条标题: {headline['headline_title']}
正文介绍: {headline['headline_paragraph']}
要点: {json.dumps(article['digest'].get('key_points', []), ensure_ascii=False)}
来源: {article['source']}
原文: {article['url']}"""

    posts = {}
    for platform, prompt in STYLE_PROMPTS.items():
        try:
            posts[platform] = _chat(prompt, content, temperature=0.7, max_tokens=1500)
        except Exception as e:
            print(f"[process] {platform} rewrite error: {e}")
            posts[platform] = ""
    return posts


# ─── Step 4: Assemble Daily Digest ──────────────────────

def assemble_digest(headline: dict, items: list[dict], cover_file: str = "") -> str:
    """Assemble the blog-style digest: 1 headline story + a numbered pick list."""
    today = datetime.now(BEIJING_TZ).strftime("%Y年%m月%d日")

    cover_section = f"\n![封面]({cover_file})\n" if cover_file else ""

    list_entries = []
    for i, it in enumerate(items, 1):
        list_entries.append(
            f"{i}. **{it['title_cn']}**（[原文]({it['url']})）— {it['blurb'].rstrip('。')}。"
        )

    digest = (
        f"# 🤖 AI DailyPulse | {today}\n\n"
        f"## 🔥 今日头条\n\n"
        f"### {headline['headline_title']}\n\n"
        f"{headline['headline_paragraph'].rstrip('。')}。\n"
        f"{cover_section}\n"
        f"[原文链接]({headline['url']})\n\n"
        f"## 📌 其他重点\n\n"
        + "\n".join(list_entries)
        + "\n\n> 📬 AI 自动生成\n"
    )

    return digest



# ─── Instagram Daily Content Pack ───────────────────────

INSTAGRAM_TOPIC_PROMPT = """Turn this verified AI-news story into one Instagram carousel content package.

Audience: people who follow AI tools, AI product updates, creator workflows, and practical AI use.
Tone: clear, premium, useful, human, not clickbait.
Language: English.

Return JSON only with:
- topic_title: short topic name
- hooks: exactly 3 strong truthful hook options
- slides: exactly 6 slides. Each slide is {"title": "...", "body": "..."}.
  Slide 1 = strongest hook/cover
  Slide 2 = what happened
  Slide 3 = what it does / key feature
  Slide 4 = why it matters
  Slide 5 = who should use/care
  Slide 6 = takeaway / CTA
- caption: Instagram-ready caption, concise and factual
- hashtags: 6 to 10 relevant hashtags without duplicates
- reel_angle: one short Reel angle / concept
- source_url: repeat the exact source URL supplied

Rules:
- No invented claims, dates, prices, features, or quotes.
- If information is uncertain, say so briefly instead of guessing.
- Keep slide text short enough for a clean carousel.
"""

def generate_instagram_pack(headline: dict, items: list[dict], articles: list[dict]) -> dict:
    by_url = {a["url"]: a for a in articles}
    selected = [{
        "url": headline["url"],
        "title": headline["headline_title"],
        "summary": headline["headline_paragraph"],
    }]
    for it in items[:4]:
        selected.append({
            "url": it["url"],
            "title": it["title_cn"],
            "summary": it["blurb"],
        })

    packs = []
    for idx, story in enumerate(selected, 1):
        article = by_url.get(story["url"], {})
        d = article.get("digest", {})
        source = article.get("source", "")
        user = (
            f"Story {idx}\n"
            f"Title: {story['title']}\n"
            f"Summary: {story['summary']}\n"
            f"Key points: {json.dumps(d.get('key_points', []), ensure_ascii=False)}\n"
            f"Source: {source}\n"
            f"Source URL: {story['url']}"
        )
        try:
            pack = _parse_json(_chat(
                INSTAGRAM_TOPIC_PROMPT,
                user,
                temperature=0.5,
                max_tokens=1800,
                json_mode=True,
            ))
        except Exception as e:
            print(f"[process] instagram pack error for {story['title'][:50]}: {e}")
            pack = {
                "topic_title": story["title"],
                "hooks": [
                    story["title"],
                    f"Here is what changed: {story['title']}",
                    f"Why this AI update matters: {story['title']}",
                ],
                "slides": [
                    {"title": story["title"], "body": ""},
                    {"title": "What happened", "body": story["summary"]},
                    {"title": "Key point", "body": (d.get("key_points") or [""])[0]},
                    {"title": "Why it matters", "body": d.get("one_liner", "")},
                    {"title": "Who should care", "body": "AI users, creators, and people following new AI tools."},
                    {"title": "Takeaway", "body": "Check the original source before trying or sharing the update."},
                ],
                "caption": f"{story['title']}\n\n{story['summary']}",
                "hashtags": ["#AI", "#AITools", "#ArtificialIntelligence", "#TechNews", "#CreatorTools", "#AIUpdates"],
                "reel_angle": f"Explain {story['title']} in under 45 seconds.",
                "source_url": story["url"],
            }

        pack["source_url"] = story["url"]
        packs.append(pack)

    payload = {
        "generated_at": datetime.now(BEIJING_TZ).isoformat(),
        "topic_count": len(packs),
        "topics": packs,
    }

    INSTAGRAM_PACK_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = [f"# Daily AI Instagram Content Pack — {datetime.now(BEIJING_TZ):%Y-%m-%d}", ""]
    for i, pack in enumerate(packs, 1):
        md.append(f"## Topic {i}: {pack.get('topic_title', '')}")
        md.append("")
        md.append("### 3 Hook Options")
        for h in (pack.get("hooks") or [])[:3]:
            md.append(f"- {h}")
        md.append("")
        md.append("### 6-Slide Carousel")
        for sidx, slide in enumerate((pack.get("slides") or [])[:6], 1):
            md.append(f"**Slide {sidx}: {slide.get('title', '')}**")
            body = slide.get("body", "")
            if body:
                md.append(body)
            md.append("")
        md.append("### Caption")
        md.append(pack.get("caption", ""))
        md.append("")
        md.append("### Hashtags")
        tags = pack.get("hashtags") or []
        md.append(" ".join(tags))
        md.append("")
        md.append("### Reel Angle")
        md.append(pack.get("reel_angle", ""))
        md.append("")
        md.append(f"Source: {pack.get('source_url', '')}")
        md.append("")
        md.append("---")
        md.append("")

    INSTAGRAM_PACK_OUTPUT.write_text("\n".join(md), encoding="utf-8")
    print(f"[process] Instagram content pack written to {INSTAGRAM_PACK_OUTPUT}")
    return payload


# ─── Orchestrator ───────────────────────────────────────

def process_all():
    """Run the full AI processing pipeline."""
    if not RAW_ARTICLES_FILE.exists():
        print("[process] no raw articles")
        return None

    articles = json.loads(RAW_ARTICLES_FILE.read_text(encoding="utf-8"))
    if not articles:
        print("[process] empty articles list")
        return None

    # Step 1: filter + summarize
    articles, errors = process_articles(articles)
    if errors and errors / len(articles) >= MAX_ERROR_RATIO:
        return {"abort": f"{errors}/{len(articles)} LLM calls failed (invalid API key or quota?)"}
    if not articles:
        print("[process] all articles filtered out")
        return None

    # Step 2: pick headline + 4 items
    final = finalize(articles)
    headline = final["headline"]
    items = final["items"]
    print(f"[process] headline: {headline['headline_title']} ({len(items)} picks)")

    # Step 3: social posts for the headline story only
    headline_article = next((a for a in articles if a["url"] == headline["url"]), articles[0])
    posts_map = {}
    if headline_article.get("digest", {}).get("relevance_score", 0) > 0.5:
        posts_map[headline_article["id"]] = generate_headline_posts(headline, headline_article)

    # Step 4: cover image from the headline article's site (never blocks the digest)
    cover_file = ""
    try:
        from cover import fetch_cover
        cover_file = fetch_cover(headline["url"]) or ""
    except Exception as e:
        print(f"[process] cover skipped: {e}")

    # Step 5: assemble blog-style digest
    digest = assemble_digest(headline, items, cover_file)
    DIGEST_OUTPUT.write_text(digest, encoding="utf-8")
    print(f"[process] digest written to {DIGEST_OUTPUT}")

    SOCIAL_OUTPUT.write_text(
        json.dumps(posts_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    instagram_pack = generate_instagram_pack(headline, items, articles)

    return {
        "digest": digest,
        "posts": posts_map,
        "instagram_pack": instagram_pack,
        "article_count": len(articles),
    }


if __name__ == "__main__":
    result = process_all()
    if result:
        print(f"\n[process] done: {result['article_count']} articles processed")