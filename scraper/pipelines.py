# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

from pathlib import Path
from urllib.parse import quote, urlparse
import json

from itemadapter import ItemAdapter

# +++ ДОБАВЬТЕ ЭТОТ КЛАСС +++
class InMemoryPipeline:
    """
    Простой pipeline, который собирает все полученные элементы (items)
    в список в памяти. Этот пайплайн будет использоваться при программном запуске.
    """
    def __init__(self):
        self.items = []

    def process_item(self, item, spider):
        self.items.append(dict(item))
        return item

    def get_items(self):
        return self.items
# +++ КОНЕЦ ДОБАВЛЕННОГО БЛОКА +++

class ScraperPipeline:
    """Pipeline that persists each scraped item to data/<sld-mod-id>.json
    - If the URL matches /system/content/doc/<mod>/<id>/ on a host like *.1gl.ru, the filename is '1gl-<mod>-<id>.json'.
    - Otherwise, falls back to percent-encoded original URL with '.json'.
    - Saves the entire item (all fields), including optional 'rank' if present, as JSON.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    @classmethod
    def from_crawler(cls, crawler):
        data_dir = crawler.settings.get("DATA_DIR", "data")
        return cls(data_dir=data_dir)

    def open_spider(self, spider):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            spider.logger.error(f"Failed to create data directory '{self.data_dir}': {e}")

    def _build_filename(self, url: str) -> str | None:
        try:
            p = urlparse(url)
            # second-level domain (e.g., '1gl' from '1gl.ru' or 'www.1gl.ru')
            host_parts = (p.netloc or "").split(":")[0].split(".")
            sld = host_parts[-2] if len(host_parts) >= 2 else (host_parts[0] if host_parts else "")
            # extract /system/content/doc/<mod>/<id>/
            path_parts = (p.path or "").strip("/").split("/")
            if len(path_parts) >= 5 and path_parts[0] == "system" and path_parts[1] == "content" and path_parts[2] == "doc":
                mod = path_parts[3]
                doc_id = path_parts[4]
                if sld and mod and doc_id:
                    return f"{sld}-{mod}-{doc_id}.json"
            return None
        except Exception:
            return None

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        # Prefer resolved_url (direct content URL) for filename derivation
        resolved_url = adapter.get("resolved_url")
        original_url = adapter.get("url") or adapter.get("original_url") or resolved_url
        if not original_url:
            spider.logger.warning("Item missing URL fields; skipping save.")
            return item

        filename = None
        url_for_pattern = resolved_url or original_url
        if url_for_pattern:
            filename = self._build_filename(url_for_pattern)
        if not filename:
            # fallback to encoded original URL
            filename = quote(str(original_url), safe="") + ".json"

        file_path = self.data_dir / filename
        try:
            data = adapter.asdict()
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            spider.logger.debug(f"Saved item to {file_path}")
        except Exception as e:
            spider.logger.error(f"Failed to write item to '{file_path}': {e}")
        return item
