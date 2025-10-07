# searchers/yandex_searcher.py

import os
import logging
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from yandex_search_api import YandexSearchAPIClient
from yandex_search_api.client import SearchType

from .base_searcher import BaseSearcher

class YandexSearcher(BaseSearcher):
    """
    Класс для поиска в Yandex, скрапинга страниц и форматирования
    результатов, полностью совместимый с архитектурой проекта.
    """

    def __init__(self, folder_id: str, oauth_token: str):
        """
        Инициализирует клиент YandexSearchAPI.

        Args:
            folder_id: Идентификатор каталога в Yandex.Cloud.
            oauth_token: OAuth-токен для авторизации.
        
        Raises:
            ValueError: Если folder_id или oauth_token не предоставлены.
        """
        if not folder_id or not oauth_token:
            raise ValueError("FOLDER_ID и OAUTH_TOKEN для YandexSearcher должны быть установлены.")
            
        self.client = YandexSearchAPIClient(folder_id=folder_id, oauth_token=oauth_token)
        logging.info("Клиент YandexSearchAPI успешно инициализирован.")

    def _scrape_page(self, url: str) -> Dict[str, str]:
        """
        Приватный метод для загрузки страницы, извлечения заголовка и текста.
        
        Args:
            url: URL-адрес для скрапинга.

        Returns:
            Словарь с 'title' и 'content' страницы.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, 'html.parser')

            # Извлекаем заголовок страницы
            title = soup.title.string if soup.title else "Без заголовка"

            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()

            content = soup.get_text(separator='\n', strip=True)
            
            return {"title": title.strip(), "content": content}

        except requests.RequestException as e:
            return {"title": "Ошибка", "content": f"Ошибка при загрузке URL {url}: {e}"}
        except Exception as e:
            return {"title": "Ошибка", "content": f"Произошла ошибка при обработке URL {url}: {e}"}

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Форматирует обработанные результаты поиска в единую строку для LLM.
        """
        if not results:
            return "Поиск в Yandex не дал результатов."

        formatted_string = ""
        for i, item in enumerate(results, 1):
            title = item.get("title", "Без заголовка")
            url = item.get("url", "Ссылка отсутствует")
            content = item.get("content", "Содержимое отсутствует")

            formatted_string += f"Источник #{i}:\n"
            formatted_string += f"  Название: {title}\n"
            formatted_string += f"  Ссылка: {url}\n"
            formatted_string += f"  Содержимое:\n\"\"\"\n{content}\n\"\"\"\n\n"
        
        return formatted_string

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
            """
            Выполняет поиск в Yandex, скрапит страницы и возвращает структурированный результат.
            """
            num_results = kwargs.get("num_results", 5)
            search_type = kwargs.get("search_type", SearchType.RUSSIAN)
            print(f"Выполняю поиск в Yandex (топ {num_results} результатов)...")

            try:
                links = self.client.get_links(
                    query_text=query,
                    search_type=search_type,
                    n_links=num_results
                )
                
                if not links:
                    print("Поиск в Yandex не вернул ссылок.")
                    return []

                processed_items = []
                for link in links:
                    if not link:
                        continue

                    print(f"  -> Yandex | Скрапинг: {link}")
                    page_data = self._scrape_page(link)
                    
                    processed_items.append({
                        "title": page_data["title"],
                        "url": link,
                        "content": page_data["content"]
                    })
                
                return processed_items

            except Exception as e:
                logging.error(f"Произошла общая ошибка при поиске в Yandex: {e}")
                return []

# --- Блок для независимого тестирования модуля ---
if __name__ == '__main__':
    import config
    from dotenv import load_dotenv

    load_dotenv()
    print("--- Тестирование модуля YandexSearcher ---")
    
    FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
    OAUTH_TOKEN = os.getenv("YANDEX_OAUTH_TOKEN")
    
    if not FOLDER_ID or not OAUTH_TOKEN:
        print("\nОшибка: Переменные окружения YANDEX_FOLDER_ID и YANDEX_OAUTH_TOKEN не установлены.")
    else:
        try:
            # searcher = YandexSearcher(folder_id=FOLDER_ID, oauth_token=OAUTH_TOKEN)
            searcher = YandexSearcher(folder_id=config.YANDEX_FOLDER_ID, 
                                        oauth_token=config.YANDEX_OAUTH_TOKEN)
            test_query = """ФСБУ 9/2025 ввели в действие в середине года.  Приведи обоснование, что компания может его применять с 1 января 2025 года, то есть задним числом. В каком периоде делать проводки по переходу: на дату принятия ФСБУ или на последнее число года? 
            Как корректировать учет с начала 2025 года до перехода? Через какие счета: 90 или 84?"""
            
            formatted_results = searcher.search(test_query, num_results=7)

            print("\n--- РЕЗУЛЬТАТ (отформатированная строка для LLM) ---")
            for d in formatted_results: 
                print(d.get("title"), d.get("url"))
            print("-------------------------------------------------")
                
        except ValueError as e:
            print(f"\nОшибка при создании экземпляра: {e}")
        except Exception as e:
            print(f"\nПроизошла непредвиденная ошибка во время теста: {e}")