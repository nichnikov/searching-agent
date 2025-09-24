import sys
from typing import List, Dict

# Предполагается, что ActionScraperRunner находится в доступном месте
# Для примера, можно положить его рядом или установить как пакет
try:
    from searchers.scraper_runner import ActionScraperRunner
except ImportError:
    print("Ошибка: Не найден модуль scraper_runner. Убедитесь, что он доступен в PYTHONPATH.")
    sys.exit(1)

from .base_searcher import BaseSearcher
from utils.formatters import format_search_results

class InternalSearcher(BaseSearcher):
    """Поисковик по внутренней базе знаний с использованием ActionScraperRunner."""

    def __init__(self, username: str, password: str):
        if not username or not password:
            raise ValueError("Логин и пароль для InternalSearcher не могут быть пустыми.")
        self.runner = ActionScraperRunner(username=username, password=password)

    def search(self, query: str, **kwargs) -> str:
        """
        Выполняет поиск во внутренней базе.

        Args:
            query: Поисковый запрос.
            **kwargs: Ожидаются 'sections' (list) и 'limit' (int).

        Returns:
            Отформатированная строка с результатами поиска.
        """
        sections = kwargs.get("sections", [])
        limit = kwargs.get("limit", 10)

        print(f"Поиск во внутренней базе по разделам: {sections}, лимит: {limit}...")
        try:
            results: List[Dict] = self.runner.search(query=query, sections=sections, limit=limit)
            return format_search_results(results, "внутренней базе знаний")
        except Exception as e:
            print(f"Произошла ошибка во время поиска во внутренней базе: {e}")
            return "Во время поиска во внутренней базе произошла ошибка."