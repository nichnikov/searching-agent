from abc import ABC, abstractmethod
from typing import Any

class BaseSearcher(ABC):
    """Абстрактный базовый класс для всех поисковых систем."""

    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> Any:
        """
        Выполняет поиск по заданному запросу.

        Args:
            query: Поисковый запрос.
            **kwargs: Дополнительные параметры для конкретной реализации.

        Returns:
            Результаты поиска в виде строки или структурированных данных.
        """
        pass