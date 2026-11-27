"""Improvement #3: collect NEWS/BLOG-style climate denial from contrarian outlets.

Fetches recent articles from known climate-contrarian sites via their RSS feeds,
scrapes the article body, and returns a denial dataframe (DenialLabel == 1) in
the same schema as the rest of the pipeline. This gives the denial class real
web-article text (not only CARDS think-tank prose), so the detector can learn
denial as it actually appears in articles.

Network-dependent and best-effort: feeds/sites change and some block scraping,
so per-feed and per-article failures are skipped rather than fatal.
"""

import re

import pandas as pd

# Known climate-contrarian outlets (RSS feeds). Best-effort; edit as needed.
CONTRARIAN_FEEDS = {
    "WUWT": "https://wattsupwiththat.com/feed/",
    "CO2Coalition": "https://co2coalition.org/feed/",
    "NoTricksZone": "https://notrickszone.com/feed/",
    "ClimateDepot": "https://www.climatedepot.com/feed/",
}

CLIMATE_KEYWORDS = re.compile(
    r"climate|warming|carbon|co2|emission|temperature|ipcc|greenhouse|"
    r"glacier|sea level|fossil|renewable",
    re.IGNORECASE,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}

STD_COLS = [
    "FullText", "Language", "DenialLabel", "DenialScore", "claim_id",
    "SourceCommonName", "DocumentIdentifier", "DATE", "Tone",
]


def _scrape(url):
    """Scrape article body text with trafilatura; '' on failure."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False,
                fast=True,
            )
            if text and len(text.strip()) > 200:
                return text.strip()
    except Exception:
        pass
    return ""


def fetch_contrarian_denial(
    max_per_source: int = 50,
    feeds=None,
    language: str = "eng",
    climate_only: bool = True,
):
    """Fetch + scrape contrarian articles and return a denial dataframe.

    Args:
        max_per_source: max feed entries to try per outlet.
        feeds: dict {source_name: rss_url}; defaults to CONTRARIAN_FEEDS.
        language: GDELT-style language tag for the rows (sources are English).
        climate_only: keep only articles whose text mentions climate terms.

    Returns:
        Pipeline-compatible denial dataframe (possibly empty if offline).
    """
    import feedparser

    feeds = feeds or CONTRARIAN_FEEDS
    texts, sources, ids = [], [], []

    for source, feed_url in feeds.items():
        try:
            parsed = feedparser.parse(feed_url, request_headers=HEADERS)
        except Exception as e:
            print(f"[contrarian] {source}: feed error ({e})")
            continue
        entries = parsed.entries[:max_per_source]
        print(f"[contrarian] {source}: {len(entries)} feed entries")
        got = 0
        for i, entry in enumerate(entries):
            link = entry.get("link")
            if not link:
                continue
            text = _scrape(link)
            if not text:
                continue
            if climate_only and not CLIMATE_KEYWORDS.search(text[:2000]):
                continue
            texts.append(text)
            sources.append(source)
            ids.append(f"contrarian:{source}:{i}")
            got += 1
        print(f"[contrarian] {source}: kept {got} climate articles")

    if not texts:
        print("[contrarian] no articles scraped (network/feeds unavailable)")
        return pd.DataFrame(columns=STD_COLS)

    n = len(texts)
    df = pd.DataFrame({
        "FullText": texts,
        "Language": language,
        "DenialLabel": 1,
        "DenialScore": 1.0,
        # each contrarian article is its own group (no translations)
        "claim_id": ids,
        "SourceCommonName": sources,
        "DocumentIdentifier": ids,
        "DATE": 0,
        "Tone": "0,0,0,0,0,0",
    })
    print(f"[contrarian] total news-style denial articles: {n} "
          f"from {df['SourceCommonName'].nunique()} source(s)")
    return df
