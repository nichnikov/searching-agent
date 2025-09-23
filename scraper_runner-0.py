# scraper_runner.py

import os
import json
import logging
from dotenv import load_dotenv
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Это по-прежнему необходимо и должно быть в самом начале
os.environ['SCRAPY_SETTINGS_MODULE'] = 'scraper.settings'

# Импортируем нашего паука напрямую
from scraper.spiders.action import ActionSpider
# Импортируем наш новый пайплайн из его правильного местоположения
from scraper.pipelines import InMemoryPipeline


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

        # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
        # Теперь мы указываем правильный, импортируемый Scrapy путь к нашему пайплайну.
        # Scrapy сможет найти 'scraper.pipelines.InMemoryPipeline'.
        # Старый файловый пайплайн будет полностью проигнорирован.
        self.settings.set('ITEM_PIPELINES', {
            'scraper.pipelines.InMemoryPipeline': 300
        })

    def search(self, query: str, sections: list[str] = None, limit: int = 5) -> list[dict]:
        """
        Запускает паука в режиме поиска и возвращает результаты.
        """
        if sections is None:
            sections = ["law", "recommendations", "forms"]

        # Создаем экземпляр нашего pipeline
        pipeline_instance = InMemoryPipeline()

        # Отключаем корневой логгер, чтобы видеть только вывод Scrapy.
        # Это решает проблемы с "зависанием" логов и делает вывод чище.
        logging.getLogger().propagate = False
        
        process = CrawlerProcess(self.settings)

        # Передаем экземпляр pipeline в паука через `crawler.signals`
        def setup_pipeline(crawler):
            crawler.pipeline = pipeline_instance

        crawler_instance = process.create_crawler(ActionSpider)
        # ВАЖНО: Мы должны вручную подключить наш pipeline к crawler'у,
        # так как Scrapy будет создавать свой собственный экземпляр по пути из настроек.
        # Этот трюк позволяет нам получить доступ к *конкретному экземпляру* после завершения.
        crawler_instance.pipeline = pipeline_instance
        
        process.crawl(
            crawler_instance,
            username=self.username,
            password=self.password,
            phrase=query,
            sections=",".join(sections),
            search_limit=limit,
        )
        
        print(f"Запускаю скрапер с запросом: '{query}'...")
        process.start() # Блокирующий вызов.
        print("Скрапер завершил работу.")

        # Возвращаем результаты из нашего экземпляра пайплайна
        return crawler_instance.pipeline.get_items()


# --- Блок для тестового запуска (остается без изменений) ---
if __name__ == '__main__':
    print("--- ЗАПУСК В ТЕСТОВОМ РЕЖИМЕ ---")
    load_dotenv()
    ACTION_USERNAME = os.getenv("ACTION_USERNAME")
    ACTION_PASSWORD = os.getenv("ACTION_PASSWORD")
    TEST_QUERY = "расчет отпускных в 2025 году"
    TEST_SECTIONS = ["recommendations"] # Уменьшил для скорости теста
    TEST_LIMIT = 2

    print(f"Тестовый запрос: '{TEST_QUERY}'")
    print(f"Разделы: {TEST_SECTIONS}, Лимит: {TEST_LIMIT}")
    print("-" * 30)

    try:
        runner = ActionScraperRunner(username=ACTION_USERNAME, password=ACTION_PASSWORD)
        results = runner.search(query=TEST_QUERY, sections=TEST_SECTIONS, limit=TEST_LIMIT)

        if results:
            print(f"\n✅ Тестовый запуск успешен! Найдено {len(results)} элементов.")
            print("--- Результаты ---")
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print("\n⚠️ Тестовый запуск завершился, но не нашел элементов. Проверьте запрос или доступность сайта.")

    except ValueError as e:
        print(f"\n❌ Ошибка конфигурации: {e}")
        print("-> Пожалуйста, убедитесь, что в файле .env заданы ACTION_USERNAME и ACTION_PASSWORD.")
    except Exception as e:
        # Добавил вывод traceback для лучшей отладки
        import traceback
        print(f"\n❌ Во время тестового запуска произошла непредвиденная ошибка: {e}")
        traceback.print_exc()