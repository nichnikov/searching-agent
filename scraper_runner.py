# scraper_runner.py

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# --- ИСПРАВЛЕНИЕ №1: РЕШЕНИЕ ПРОБЛЕМЫ ModuleNotFoundError ---
# Добавляем корневую папку проекта в sys.path.
# Это гарантирует, что Python и Scrapy смогут найти модуль 'scraper'.
# Это действие должно быть выполнено ДО импорта компонентов Scrapy.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Теперь, когда sys.path исправлен, эти импорты будут работать
os.environ['SCRAPY_SETTINGS_MODULE'] = 'scraper.settings'
from scraper.spiders.action import ActionSpider


class ActionScraperRunner:
    """
    Класс-обертка для программного запуска паука ActionSpider.
    """
    def __init__(self, username: str, password: str):
        if not username or not password:
            raise ValueError("Имя пользователя и пароль не могут быть пустыми.")
        self.username = username
        self.password = password
        self.settings = get_project_settings()

        # Мы по-прежнему указываем Scrapy использовать наш пайплайн
        self.settings.set('ITEM_PIPELINES', {
            'scraper.pipelines.InMemoryPipeline': 300
        })

    def search(self, query: str, sections: list[str] = None, limit: int = 5) -> list[dict]:
        """
        Запускает паука в режиме поиска и возвращает результаты.
        """
        # --- ИСПРАВЛЕНИЕ №2: НАДЕЖНЫЙ СБОР РЕЗУЛЬТАТОВ ЧЕРЕЗ СИГНАЛЫ ---
        results = []

        def item_scraped(item):
            """Эта функция будет вызываться каждый раз, когда паук выдает item."""
            results.append(dict(item))

        # Отключаем корневой логгер, чтобы видеть только вывод Scrapy
        logging.getLogger().propagate = False
        
        process = CrawlerProcess(self.settings)
        
        # Подключаем нашу функцию к сигналу item_scraped
        crawler = process.create_crawler(ActionSpider)
        crawler.signals.connect(item_scraped, signal=signals.item_scraped)
        
        process.crawl(
            crawler,
            username=self.username,
            password=self.password,
            phrase=query,
            sections=",".join(sections),
            search_limit=limit,
        )
        
        print(f"Запускаю скрапер с запросом: '{query}'...")
        process.start()  # Блокирующий вызов.
        print("Скрапер завершил работу.")

        return results


# --- Блок для тестового запуска (без изменений) ---
if __name__ == '__main__':
    print("--- ЗАПУСК В ТЕСТОВОМ РЕЖИМЕ ---")
    load_dotenv()
    ACTION_USERNAME = os.getenv("ACTION_USERNAME")
    ACTION_PASSWORD = os.getenv("ACTION_PASSWORD")
    
    TEST_QUERY = "Добрый день! Такой вопрос. У нас ИП на НДС. У нас с нового года в кассовых чеках при оплате наличными от юр.лиц пишется без НДС. Это правильно или так не должно быть."
    
    TEST_SECTIONS = ["recommendations"]
    TEST_LIMIT = 20

    print(f"Тестовый запрос: '{TEST_QUERY}'")
    print(f"Разделы: {TEST_SECTIONS}, Лимит: {TEST_LIMIT}")
    print("-" * 30)

    try:
        runner = ActionScraperRunner(username=ACTION_USERNAME, password=ACTION_PASSWORD)
        results = runner.search(query=TEST_QUERY, sections=TEST_SECTIONS, limit=TEST_LIMIT)

        if results:
            print("--- Результаты ---")
            print(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"\n✅ Тестовый запуск успешен! Найдено {len(results)} элементов.")
        else:
            print("\n⚠️ Тестовый запуск завершился, но не нашел элементов. Проверьте запрос или доступность сайта.")

    except ValueError as e:
        print(f"\n❌ Ошибка конфигурации: {e}")
        print("-> Пожалуйста, убедитесь, что в файле .env заданы ACTION_USERNAME и ACTION_PASSWORD.")
    
    except Exception as e:
        
        import traceback
        print(f"\n❌ Во время тестового запуска произошла непредвиденная ошибка: {e}")
        traceback.print_exc()
