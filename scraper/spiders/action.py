from urllib.parse import urljoin, urlparse, quote_plus

import scrapy
from scrapy_playwright.page import PageMethod

BLOCK_LINK_SUBSTRINGS = ("mc.yandex", "metrika.yandex")

STATIC_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".css", ".woff", ".woff2")

class ActionSpider(scrapy.Spider):
    name = "action"
    login_url = "https://id2.action-media.ru/Logon"

    def __init__(
        self,
        username: str,
        password: str,
        page_url: str | None = None,
        traverse_search: bool = False,
        phrase: str | None = None,
        search_limit: int = 10,
        sections: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.username = username
        self.password = password
        self.page_url = page_url
        self.traverse_search = bool(traverse_search)
        self.phrase = phrase
        self.search_limit = int(search_limit or 0) if search_limit is not None else 0
        default_sections = ["recommendations", "law", "forms", "handbook", "press"]
        self.sections = [s.strip().lower() for s in sections.split(",") if s.strip()] if sections else default_sections

        # If phrase is provided, force search traversal regardless of traverse_search flag
        if self.phrase:
            self.traverse_search = True
        
        # cross-section dedup storage
        self._seen_urls = set()
        self._seen_direct_urls = set()

    def _to_direct_content_url(self, url: str) -> str:
        """Transform SPA hash URL to direct content URL.
        Example:
        https://1jur.ru/#/document/98/103491141 -> https://1jur.ru/system/content/doc/98/103491141/
        """
        try:
            parsed = urlparse(url)
            fragment = parsed.fragment or ""
            candidate = fragment.lstrip("/")
            if not candidate:
                p = parsed.path or ""
                if p.startswith("/#/"):
                    candidate = p[3:].lstrip("/")
                elif p.startswith("/document/"):
                    candidate = p.lstrip("/")
            if candidate.startswith("document/"):
                parts = candidate.split("/")
                if len(parts) >= 3:
                    mod, doc_id = parts[1], parts[2]
                    return f"{parsed.scheme}://{parsed.netloc}/system/content/doc/{mod}/{doc_id}/"
            return url
        except Exception:
            return url

    def _build_search_urls(self, phrase: str, sections: list[str]) -> list[str]:
        encoded = quote_plus(phrase)
        mapping = {
            "recommendations": f"https://1gl.ru/?#/recommendations/found/fixedregioncode=all&ishiddensearch=false&isusehints=false&phrase={encoded}&sort=Relevance/",
            "law": f"https://1gl.ru/?#/law/found/clearregions=true&ishiddensearch=false&isusehints=false&phrase={encoded}&sort=Relevance/",
            "forms": f"https://1gl.ru/?#/forms/found/fixedregioncode=all&ishiddensearch=false&isusehints=false&phrase={encoded}&sort=Relevance/",
            "handbook": f"https://1gl.ru/?#/handbook/found/fixedregioncode=all&ishiddensearch=false&isusehints=false&phrase={encoded}&sort=Relevance/",
            "press": f"https://1gl.ru/?#/press/found/clearregions=true&ishiddensearch=false&isusehints=false&phrase={encoded}&sort=Relevance/",
        }
        urls = []
        for s in sections:
            if s in mapping:
                urls.append(mapping[s])
        return urls

    def _route_blocking(self, route):
        try:
            req = route.request
            url = (getattr(req, "url", "") or "").lower()
            # Block analytics
            if any(s in url for s in BLOCK_LINK_SUBSTRINGS):
                return route.abort()
            # Block auth_check polling that prevents network idle
            if "/auth/check-backend/api/v2/auth_check" in url or "/auth_check" in url:
                return route.abort()
            # Block by resource type
            rtype = getattr(req, "resource_type", None)
            if rtype in {"image", "media", "font", "stylesheet"}:
                return route.abort()
            # Block by extension
            url_no_qs = url.split("?", 1)[0]
            if url_no_qs.endswith(STATIC_EXTS):
                return route.abort()
            return route.continue_()
        except Exception:
            return route.continue_()

    def start_requests(self):
        yield scrapy.Request(
            self.login_url,
            meta={
                "playwright": True,
                "playwright_context": "default",
                "playwright_page_methods": [
                    PageMethod(
                        "route",
                        "**/*",
                        self._route_blocking,
                    ),
                    PageMethod("fill", 'input[data-qa-locator="login"]', self.username),
                    PageMethod("fill", 'input[data-qa-locator="password"]', self.password),
                    PageMethod("click", 'button[data-qa-locator="submit"]'),
                    PageMethod("wait_for_load_state", "domcontentloaded"),
                    PageMethod("wait_for_timeout", 800),
                ],
            },
            dont_filter=True,
            callback=self.after_login,
        )

    async def after_login(self, response):
        if response.xpath('//input[@data-qa-locator="login"]').get():
            self.logger.error("Login failed (still on login page). Check credentials or 2FA/CAPTCHA.")
            return

        # Decide mode: direct page vs search
        if self.page_url:
            self.logger.info("Login succeeded. Scheduling direct page parse: 1 URL")
            url = self.page_url
            transformed = self._to_direct_content_url(url)
            yield scrapy.Request(
                transformed,
                meta={
                    "playwright": True,
                    "playwright_context": "default",
                    "original_url": url,
                    "playwright_page_methods": [
                        PageMethod("route", "**/*", self._route_blocking),
                        PageMethod("evaluate", "async () => { await new Promise(r => setTimeout(r, 600)); }"),
                        PageMethod("wait_for_load_state", "domcontentloaded", timeout=30000),                      ],
                },
                callback=self.parse_page,
                dont_filter=True,
            )
            return

        if self.phrase and self.sections:
            search_urls = self._build_search_urls(self.phrase, self.sections)
            self.logger.info("Login succeeded. Scheduling search in %d section(s): %d URL(s)", len(self.sections), len(search_urls))
            for url in search_urls:
                yield scrapy.Request(
                    url,
                    meta={
                        "playwright": True,
                        "playwright_context": "default",
                        "playwright_page_methods": [
                            PageMethod("route", "**/*", self._route_blocking),
                            PageMethod("wait_for_selector", "div[data-id='search-results-section']", timeout=30000),
                            PageMethod("wait_for_selector", "div[data-id='search-item']", timeout=30000),
                            PageMethod(
                                "evaluate",
                                "async () => {\n  const sel = 'div[data-id=\\'search-item\\']';\n  for (let i = 0; i < 5; i++) {\n    const before = document.querySelectorAll(sel).length;\n    window.scrollTo(0, document.body.scrollHeight);\n    await new Promise(r => setTimeout(r, 1200));\n    const after = document.querySelectorAll(sel).length;\n    if (after <= before) break;\n  }\n}",
                            ),
                            PageMethod("wait_for_timeout", 800),
                        ],
                    },
                    callback=self.parse_search,
                    dont_filter=True,
                )
            return

        self.logger.warning("Nothing to do: provide either page_url for direct parsing or phrase with sections for search.")

    def parse_search(self, response):
        base = f"{urlparse(response.url).scheme}://{urlparse(response.url).netloc}"

        items = response.css("div[data-id='search-item']")
        if not items:
            self.logger.warning("No result items found on %s", response.url)

        seen_local = set()
        sent = 0
        limit = self.search_limit if (self.search_limit and self.search_limit > 0) else None

        for idx, it in enumerate(items, start=1):
            href = it.css("div[data-qa-locator='title'] a::attr(href)").get()
            title = (it.css("div[data-qa-locator='title'] a::text").get() or "").strip()
            description = (it.css("div[data-qa-locator='description']::text").get() or "").strip()

            if not href:
                href = it.css("a[href*='#/document/']::attr(href)").get()
            if not href:
                continue

            full_spa = urljoin(base + "/", href.lstrip("/"))
            if full_spa in seen_local or full_spa in self._seen_urls:
                continue

            direct = self._to_direct_content_url(full_spa)
            if direct in self._seen_direct_urls:
                continue

            seen_local.add(full_spa)
            self._seen_urls.add(full_spa)
            self._seen_direct_urls.add(direct)

            yield scrapy.Request(
                direct,
                meta={
                    "playwright": True,
                    "playwright_context": "default",
                    "original_url": full_spa,
                    "_search_title": title,
                    "_search_description": description,
                    "rank": idx,
                    "playwright_page_methods": [
                        PageMethod("route", "**/*", self._route_blocking),
                        PageMethod("evaluate", "async () => { await new Promise(r => setTimeout(r, 600)); }"),
                        PageMethod("wait_for_load_state", "domcontentloaded", timeout=30000),
                    ],
                },
                callback=self.parse_page,
                dont_filter=True,
            )
            sent += 1
            if limit and sent >= limit:
                break

    def parse_page(self, response):
        title = (
            response.xpath("normalize-space(//h1//text())").get()
            or response.xpath("normalize-space(//title//text())").get()
            or ""
        )

        content_root = response.xpath("//div[@data-name='page|contentHTML']")
        # Plain text fallback
        texts = content_root.xpath(
            ".//text()[normalize-space() and not(ancestor::script or ancestor::style or ancestor::noscript)]"
        ).getall() or response.xpath(
            "//body//text()[normalize-space() and not(ancestor::script or ancestor::style or ancestor::noscript)]"
        ).getall()

        clean = [t.strip() for t in texts if t and t.strip()]
        content = " ".join(clean)

        original_url = response.meta.get("original_url") or response.url

        yield {
            "rank": response.meta.get("rank"),
            "url": original_url,
            "resolved_url": response.url,
            "title": title,
            "content": content,
        }
