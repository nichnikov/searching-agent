from typing import List, Dict, Any, Set
from abc import ABC, abstractmethod

# --- Базовые и вспомогательные классы (для примера) ---

class BaseSearcher(ABC):
    """Абстрактный базовый класс для всех поисковиков."""
    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        pass

class GoogleSearcher(BaseSearcher):
    """Мок-класс (заглушка) для имитации поиска в Google."""
    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        print(f"-> [Google] Поиск по запросу: '{query}'")
        # Имитация результатов
        if "python" in query:
            return [
                {"title": "Welcome to Python.org", "url": "https://www.python.org/", "snippet": "The official home of the Python Programming Language."},
                {"title": "Real Python: Python Tutorials", "url": "https://realpython.com/", "snippet": "Learn Python online: Python tutorials for developers of all skill levels..."},
            ]
        return []

class YandexSearcher(BaseSearcher):
    """Мок-класс (заглушка) для имитации поиска в Яндексе."""
    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        print(f"-> [Yandex] Поиск по запросу: '{query}'")
        # Имитация результатов с одним повторяющимся URL
        if "python" in query:
            return [
                {"title": "Курсы Python-разработчик от Яндекс.Практикум", "url": "https://practicum.yandex.ru/python/", "snippet": "Станьте Python-разработчиком с нуля."},
                {"title": "Real Python - Обучение Python", "url": "https://realpython.com/", "snippet": "Статьи и уроки по языку Python."},
            ]
        return []

# --- Ваш класс CombinedWebSearcher (без изменений) ---

class CombinedWebSearcher(BaseSearcher):
    def __init__(self, searchers: List[BaseSearcher]):
        if not searchers:
            raise ValueError("Список поисковиков не может быть пустым.")
        self.searchers = searchers
        print(f"Комбинированный поисковик инициализирован с {len(self.searchers)} провайдерами.")

    def _add_if_unique(self, item: Dict[str, Any], all_results: List[Dict[str, Any]], seen_urls: Set[str]) -> None:
        url = item.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_results.append(item)

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        all_results = []
        seen_urls = set()
        for searcher in self.searchers:
            provider_name = type(searcher).__name__ 
            try:
                results = searcher.search(query, **kwargs)
                for item in results:
                    self._add_if_unique(item, all_results, seen_urls)
            except Exception as e:
                print(f"Ошибка при поиске через {provider_name}: {e}")
        print(f"Объединенный поиск по '{query}' дал {len(all_results)} уникальных результатов.")
        return all_results

# --- Новый класс для мульти-поиска ---

class MultiQuerySearcher:
    """
    Класс для выполнения серии поисковых запросов через CombinedWebSearcher
    и агрегации всех результатов в единый уникальный список.
    """
    def __init__(self, combined_searcher: CombinedWebSearcher):
        """
        Инициализирует мульти-поисковик экземпляром CombinedWebSearcher.
        
        Args:
            combined_searcher: Готовый к работе экземпляр CombinedWebSearcher.
        """
        if not isinstance(combined_searcher, CombinedWebSearcher):
            raise TypeError("Необходимо передать экземпляр CombinedWebSearcher.")
        self.combined_searcher = combined_searcher
        print("Мульти-поисковик инициализирован.")

    def _add_if_unique(self, item: Dict[str, Any], all_results: List[Dict[str, Any]], seen_urls: Set[str]) -> None:
        """
        Добавляет результат в общий список, если его URL уникален.
        Этот метод нужен для дедупликации результатов *между* разными запросами.
        """
        url = item.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_results.append(item)

    def search_all(self, queries: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Выполняет поиск по каждому запросу из списка и возвращает
        единый дедуплицированный список результатов.
        
        Args:
            queries: Список поисковых запросов (e.g., ["python уроки", "python для начинающих"]).
            
        Returns:
            Единый список словарей с уникальными результатами.
        """
        aggregated_results = []
        seen_urls = set() # Множество для отслеживания URL на протяжении всех запросов

        print(f"\nНачинаем мульти-поиск по {len(queries)} запросам...")
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- [Шаг {i}/{len(queries)}] Обработка запроса: '{query}' ---")
            
            # Используем CombinedWebSearcher для получения уникальных результатов для ОДНОГО запроса
            results_for_current_query = self.combined_searcher.search(query, **kwargs)
            
            # Добавляем полученные результаты в общий список,
            # проверяя на уникальность уже среди ВСЕХ найденных результатов.
            for item in results_for_current_query:
                self._add_if_unique(item, aggregated_results, seen_urls)
                
        print(f"\n✅ Мульти-поиск завершен. Найдено {len(aggregated_results)} уникальных результатов по всем запросам.")
        return aggregated_results

# --- Пример использования ---

if __name__ == "__main__":
    # 1. Создаем экземпляры поисковиков
    google_searcher = GoogleSearcher()
    yandex_searcher = YandexSearcher()
    
    # 2. Создаем комбинированный поисковик
    combined_searcher = CombinedWebSearcher([google_searcher, yandex_searcher])
    
    # 3. Создаем мульти-поисковик, передавая ему комбинированный
    multi_searcher = MultiQuerySearcher(combined_searcher)
    
    # 4. Определяем список близких по смыслу запросов
    search_queries = [
        "python for beginners",
        "learn python"
    ]
    
    # 5. Запускаем поиск по всем запросам
    final_unique_results = multi_searcher.search_all(search_queries)
    
    # 6. Выводим итоговый результат
    print("\n--- Итоговый уникальный список результатов ---")
    for idx, result in enumerate(final_unique_results, 1):
        print(f"{idx}. {result['title']} ({result['url']})")