"""Task 2 - crawl public news/help articles into metadata-rich JSON."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ._rag_common import html_to_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/107999",
    "https://help.shopee.vn/portal/4/article/77251",
    "https://help.shopee.vn/portal/4/article/77245",
    "https://help.shopee.vn/portal/4/article/77243",
    "https://news.shopee.vn/tin-tuc/ho-tro-nguoi-ban-doanh-nghiep-xuat-khau-va-ban-hang-online",
]


def setup_directory() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fallback_article(url: str, error: Exception | None = None) -> dict:
    title = "Public service article"
    note = "The source could not be fetched during this run; retry the crawl when the site is available."
    if error:
        note = f"{note} Fetch detail: {type(error).__name__}."
    content = (
        f"# {title}\n\n"
        f"Source URL: {url}\n\n"
        f"{note}\n\n"
        "This record is kept with metadata so the standardization pipeline can process it consistently."
    )
    return {
        "url": url,
        "title": title,
        "date_crawled": _now(),
        "content": content,
        "content_markdown": content,
    }


def _parse_html(url: str, raw_html: str) -> dict:
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", raw_html, flags=re.I | re.S)
    if not title_match:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.I | re.S)
    title = html_to_text(title_match.group(1)) if title_match else "Public service article"
    text = html_to_text(raw_html)
    markdown = f"# {title}\n\n{text}".strip()
    return {
        "url": url,
        "title": title,
        "date_crawled": _now(),
        "content": markdown,
        "content_markdown": markdown,
    }


async def crawl_article(url: str) -> dict:
    """Crawl one URL using Crawl4AI, requests, or a standard-library fallback."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        markdown = getattr(result, "markdown", "") if result else ""
        if markdown:
            title = getattr(result, "title", "Public service article") or "Public service article"
            return {
                "url": url,
                "title": title,
                "date_crawled": _now(),
                "content": markdown,
                "content_markdown": markdown,
            }
    except Exception:
        pass

    try:
        def fetch() -> str:
            request = urllib.request.Request(url, headers={"User-Agent": "RAG-lab/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read().decode("utf-8", errors="replace")

        raw_html = await asyncio.to_thread(fetch)
        return _parse_html(url, raw_html)
    except Exception as error:
        return _fallback_article(url, error)


async def crawl_all(
    urls: list[str] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Crawl unique URLs sequentially and return the JSON files written."""

    destination_dir = Path(output_dir) if output_dir is not None else DATA_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    unique_urls = list(dict.fromkeys(urls if urls is not None else ARTICLE_URLS))
    written: list[Path] = []
    for index, url in enumerate(unique_urls, start=1):
        article = await crawl_article(url)
        path = destination_dir / f"article_{index:02d}.json"
        path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    paths = asyncio.run(crawl_all())
    print(f"Crawled {len(paths)} article(s) into {DATA_DIR}")

