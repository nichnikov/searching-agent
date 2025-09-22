import os
import json
import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import GoogleSerperAPIWrapper

class WebSearcher:
    """
    Класс для выполнения поиска в Интернете с использованием Serper API
    и извлечения текста с веб-страниц.
    """
    def __init__(self, api_key=None):
        """
        Инициализирует WebSearcher с вашим ключом API Serper.
        :param api_key: Ваш ключ API Serper. Если не предоставлен,
                        будет предпринята попытка получить его из переменной окружения SERPER_API_KEY.
        """
        if api_key:
            os.environ["SERPER_API_KEY"] = api_key
        elif not os.getenv("SERPER_API_KEY"):
            raise ValueError("Необходимо предоставить SERPER_API_KEY или установить его в качестве переменной окружения.")

        self.search = GoogleSerperAPIWrapper()

    def _scrape_text_from_url(self, url: str) -> str:
        """
        Загружает и извлекает весь видимый текст с веб-страницы.
        :param url: URL-адрес для извлечения текста.
        :return: Извлеченный текст или сообщение об ошибке.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text

        except requests.RequestException as e:
            return f"Ошибка при загрузке URL: {e}"
        except Exception as e:
            return f"Произошла ошибка при обработке: {e}"

    def search_internet(self, query: str, num_results: int = 5) -> dict:
        """
        Выполняет поиск по заданному запросу и извлекает текст с найденных страниц.
        :param query: Поисковый запрос.
        :param num_results: Количество результатов для обработки.
        :return: Словарь с результатами поиска.
        """
        try:
            search_results = self.search.results(query)

            if "organic" not in search_results:
                return {"error": "Не удалось найти органические результаты."}

            processed_results = {
                "searchParameters": search_results.get("searchParameters", {}),
                "organic": []
            }

            for item in search_results["organic"][:num_results]:
                link = item.get("link")
                scraped_text = self._scrape_text_from_url(link) if link else "Ссылка отсутствует."
                item["full_text"] = scraped_text
                processed_results["organic"].append(item)

            return processed_results

        except Exception as e:
            return {"error": f"Произошла общая ошибка: {e}"}

    def save_results_to_json(self, results: dict, filename: str = "search_results.json"):
        """
        Сохраняет словарь с результатами в файл JSON.
        :param results: Словарь с результатами для сохранения.
        :param filename: Имя файла для сохранения.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Результаты успешно сохранены в файл {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении в JSON: {e}")


if __name__ == '__main__':
    # Пример использования класса
    # Установите ваш API-ключ здесь или в переменных окружения
    # SERPER_API_KEY = "ВАШ_КЛЮЧ_API"
    # searcher = WebSearcher(api_key=SERPER_API_KEY)

    try:
        # Инициализация без прямого указания ключа (с использованием переменных окружения)
        searcher = WebSearcher()

        query = "сотрудник идет в отпуск в январе 2025 расчетный период 2024 год Средневой составил 680,48 нужно ли учитывать до МРОТА 2025"
        results = searcher.search_internet(query, num_results=3)

        # Вывод результатов
        print(json.dumps(results, ensure_ascii=False, indent=4))

        # Сохранение результатов в файл
        if "error" not in results:
            searcher.save_results_to_json(results)

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")