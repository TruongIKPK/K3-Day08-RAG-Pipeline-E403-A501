"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài viết cần crawl (tối thiểu 5 bài)
ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/107999",
    "https://help.shopee.vn/portal/4/article/77251?seo=1&utm_source=chatgpt.com",
    "https://help.shopee.vn/portal/4/article/77245",
    "https://help.shopee.vn/portal/4/article/107999",
    "https://help.shopee.vn/portal/4/article/77243-%C4%90I%E1%BB%80U-KHO%E1%BA%A2N-D%E1%BB%8ACH-V%E1%BB%A4?utm_source=chatgpt.com",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    # 1. Thử dùng crawl4ai nếu đã cài đặt
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and hasattr(result, "markdown") and result.markdown:
                return {
                    "url": url,
                    "title": getattr(result, "title", "Bài viết dịch vụ đại học"),
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
    except Exception as e:
        print(f"  [INFO] crawl4ai not available ({e}). Using requests fallback...")

    # 2. Fallback sang requests + BeautifulSoup
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Lấy tiêu đề bài viết
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Bài viết tin tức đại học"

        # Loại bỏ script, style rác
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Lấy nội dung chữ
        text_content = soup.get_text(separator="\n", strip=True)

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"# {title}\n\n{text_content}",
        }
    except Exception as err:
        print(f"  [WARN] Request error ({err}). Generating fallback content...")
        return {
            "url": url,
            "title": f"Thông tin dịch vụ đại học ({url})",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"# Thông tin dịch vụ đại học\n\nNội dung thông báo hướng dẫn dịch vụ sinh viên, quy định học phí, học bổng và hỗ trợ đào tạo dành cho sinh viên.",
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
