from typing import List, Dict, Any
from .base_searcher import BaseSearcher

class CombinedWebSearcher(BaseSearcher):
    """
    Класс-агрегатор для выполнения поиска в нескольких внешних системах,
    объединения результатов и их дедупликации.
    """
    def __init__(self, searchers: List[BaseSearcher]):
        """
        Инициализирует агрегатор списком экземпляров поисковиков.
        
        Args:
            searchers: Список объектов, унаследованных от BaseSearcher
                       (например, [WebSearcher(), YandexSearcher()]).
        """
        if not searchers:
            raise ValueError("Список поисковиков не может быть пустым.")
        self.searchers = searchers
        print(f"Комбинированный поисковик инициализирован с {len(self.searchers)} провайдерами.")

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Выполняет поиск во всех провайдерах, объединяет и дедуплицирует результаты.
        """
        all_results = []
        seen_urls = set()

        for searcher in self.searchers:
            # Каждый поисковик может иметь свое имя класса (WebSearcher, YandexSearcher)
            provider_name = type(searcher).__name__ 
            try:
                results = searcher.search(query, **kwargs)
                for item in results:
                    url = item.get("url")
                    # Проверяем наличие URL и его уникальность
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(item)
            except Exception as e:
                print(f"Ошибка при поиске через {provider_name}: {e}")
        
        print(f"Объединенный поиск дал {len(all_results)} уникальных результатов.")
        return all_results