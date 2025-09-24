from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BaseSearcher(ABC):
    """Абстрактный базовый класс для всех поисковых систем."""

    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Выполняет поиск по заданному запросу.

        Args:
            query: Поисковый запрос.
            **kwargs: Дополнительные параметры для конкретной реализации.

        Returns:
            Список словарей, где каждый словарь представляет найденный
            материал и содержит как минимум ключи 'url', 'title', 'content'.
        """
        pass