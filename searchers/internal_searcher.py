import sys
from typing import List, Dict,  Any

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

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Выполняет поиск во внутренней базе и возвращает структурированный результат.
        """
        sections = kwargs.get("sections", [])
        limit = kwargs.get("limit", 10)

        print(f"Поиск во внутренней базе по разделам: {sections}, лимит: {limit}...")
        try:
            results: List[Dict] = self.runner.search(query=query, sections=sections, limit=limit)
            return results
        except Exception as e:
            print(f"Произошла ошибка во время поиска во внутренней базе: {e}")
            return [] # Возвращаем пустой список в случае ошибки
        



# --- Блок для независимого тестирования модуля ---
if __name__ == '__main__':
    import os
    import json

    # --- Подготовка для запуска из командной строки ---
    # Этот хак позволяет Python найти родительские модули (например, config),
    # когда скрипт запускается напрямую.
    # Добавляем корневую папку проекта в путь поиска модулей.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    
    # Теперь можно импортировать config
    import config

    print("-" * 50)
    print("--- Тестирование модуля InternalSearcher ---")
    print("-" * 50)

    # Проверяем, что учетные данные загружены из .env через config
    if not config.ACTION_USERNAME or not config.ACTION_PASSWORD:
        print("\n[ОШИБКА] Не найдены учетные данные.")
        print("Пожалуйста, убедитесь, что в файле .env заданы ACTION_USERNAME и ACTION_PASSWORD.")
    else:
        try:
            # 1. Инициализация поисковика
            print("\nШаг 1: Инициализация InternalSearcher...")
            searcher = InternalSearcher(
                username=config.ACTION_USERNAME,
                password=config.ACTION_PASSWORD
            )
            
            # 2. Определение тестового запроса
            test_query = config.DEFAULT_QUERY
            test_sections = config.DEFAULT_SECTIONS.split(',')
            test_limit = 3 # Используем небольшой лимит для теста
            
            print(f"\nШаг 2: Выполнение тестового поиска (лимит: {test_limit})...")
            print(f"Запрос: '{test_query[:80]}...'")

            # 3. Выполнение поиска
            results = searcher.search(
                query=test_query,
                sections=test_sections,
                limit=test_limit
            )

            # 4. Вывод результатов
            print("\nШаг 3: Анализ результатов...")
            if results:
                print(f"✅ Успех! Найдено результатов: {len(results)}")
                print("Структура первого результата:")
                # Используем json.dumps для красивого вывода словаря
                print(json.dumps(results[0], ensure_ascii=False, indent=2))
            else:
                print("⚠️ Поиск завершился, но не вернул результатов.")
                print("Это может быть нормально, если по запросу ничего не нашлось, либо произошла ошибка (см. логи выше).")

        except Exception as e:
            print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Во время теста произошла ошибка: {e}")

    print("\n--- Тестирование завершено ---")