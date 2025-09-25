import os
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain_community.utilities import GoogleSerperAPIWrapper

from .base_searcher import BaseSearcher

class WebSearcher(BaseSearcher):
    """
    Класс для поиска в интернете с использованием Serper API и скрапинга
    текста со страниц. Адаптирован для архитектуры проекта.
    """

    def __init__(self, api_key: str = None):
        """
        Инициализирует WebSearcher с API-ключом Serper.

        Args:
            api_key: Ключ Serper API. Если не указан, будет использована
                     переменная окружения SERPER_API_KEY.
        
        Raises:
            ValueError: Если ключ не предоставлен и не найден в переменных окружения.
        """
        # Используем предоставленный ключ или получаем из окружения
        effective_api_key = api_key or os.getenv("SERPER_API_KEY")
        if not effective_api_key:
            raise ValueError("Необходимо предоставить SERPER_API_KEY или установить его в качестве переменной окружения.")
        
        # Устанавливаем ключ в окружение для langchain_community
        os.environ["SERPER_API_KEY"] = effective_api_key
        self.search_wrapper = GoogleSerperAPIWrapper()

    def _scrape_text_from_url(self, url: str) -> str:
        """
        Приватный метод для загрузки и извлечения видимого текста с веб-страницы.
        
        Args:
            url: URL-адрес для скрапинга.

        Returns:
            Извлеченный текст или сообщение об ошибке.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()  # Проверка на HTTP ошибки
            response.encoding = response.apparent_encoding # Улучшаем обработку кодировок

            soup = BeautifulSoup(response.text, 'html.parser')

            # Удаляем все теги script и style
            for element in soup(["script", "style", "header", "footer", "nav"]):
                element.decompose()

            text = soup.get_text(separator='\n', strip=True)
            return text

        except requests.RequestException as e:
            return f"Ошибка при загрузке URL {url}: {e}"
        except Exception as e:
            return f"Произошла ошибка при обработке URL {url}: {e}"

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Форматирует обработанные результаты поиска в единую строку для LLM.
        Эта функция аналогична `format_search_results` в `utils/formatters.py`
        для сохранения консистентности вывода.
        """
        if not results:
            return "Поиск в интернете не дал результатов."

        formatted_string = ""
        for i, item in enumerate(results, 1):
            title = item.get("title", "Без заголовка")
            url = item.get("url", "Ссылка отсутствует")
            content = item.get("content", "Содержимое отсутствует")
            
            # Обрезаем контент для экономии токенов
            formatted_string += f"Источник #{i}:\n"
            formatted_string += f"  Название: {title}\n"
            formatted_string += f"  Ссылка: {url}\n"
            formatted_string += f"  Содержимое:\n\"\"\"\n{content}\n\"\"\"\n\n"
        
        return formatted_string

    def search(self, query: str, **kwargs: Any) -> str:
        """
        Выполняет поиск по запросу, скрапит страницы и возвращает отформатированную строку.

        Args:
            query: Поисковый запрос.
            **kwargs: Поддерживаемый параметр 'num_results' (int) - количество результатов для обработки.

        Returns:
            Единая строка с отформатированными результатами поиска.
        """
        num_results = kwargs.get("num_results", 5)
        print(f"Выполняю поиск в интернете (топ {num_results} результатов)...")

        try:
            # Устанавливаем количество результатов через атрибут 'k' перед вызовом
            self.search_wrapper.k = num_results
            search_results = self.search_wrapper.results(query)
            print(f"Найдено результатов: {len(search_results.get('organic', []))}")

            if "organic" not in search_results or not search_results["organic"]:
                return "Поиск в интернете не дал органических результатов."

            processed_items = []
            for item in search_results["organic"][:num_results]:
                link = item.get("link")
                if not link:
                    continue

                print(f"  -> Скрапинг: {link}")
                scraped_text = self._scrape_text_from_url(link)
                
                # Собираем данные в унифицированном формате
                processed_items.append({
                    "title": item.get("title", "Без заголовка"),
                    "url": link,
                    "content": scraped_text
                })
            
            return processed_items

        except Exception as e:
            error_message = f"Произошла общая ошибка при поиске в интернете: {e}"
            print(error_message)
            return error_message


# --- Блок для независимого тестирования модуля ---
if __name__ == '__main__':
    from dotenv import load_dotenv
    import json

    # Загружаем переменные из .env для теста
    load_dotenv()

    print("--- Тестирование модуля WebSearcher ---")
    try:
        # Инициализация с использованием переменной окружения SERPER_API_KEY
        searcher = WebSearcher()
        
        test_query = "кто президент Эфиопии в 2010 году"
        
        # Вызов основного метода search
        formatted_string_results = searcher.search(test_query, num_results=15)

        print("\n--- РЕЗУЛЬТАТ (отформатированная строка для LLM) ---")
        # print(formatted_string_results)
        print("-------------------------------------------------")

    except ValueError as e:
        print(f"Ошибка инициализации: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка во время теста: {e}")