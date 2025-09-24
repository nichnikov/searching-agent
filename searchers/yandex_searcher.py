import os
import logging
from yandex_search_api import YandexSearchAPIClient
from yandex_search_api.client import SearchType

class YandexSearcher:
    """
    Класс для взаимодействия с Yandex Search API.

    Этот класс инкапсулирует логику инициализации клиента
    и выполнения поисковых запросов.
    """
    def __init__(self, folder_id: str, oauth_token: str):
        """
        Инициализирует клиент YandexSearchAPI.

        Args:
            folder_id (str): Идентификатор вашего каталога в Yandex.Cloud.
            oauth_token (str): Ваш OAuth-токен для авторизации.
        """
        if not folder_id or not oauth_token:
            raise ValueError("FOLDER_ID и OAUTH_TOKEN должны быть установлены.")
            
        self.client = YandexSearchAPIClient(
            folder_id=folder_id,
            oauth_token=oauth_token
        )
        logging.info("Клиент YandexSearchAPI успешно инициализирован.")

    def search(self, query_text: str, search_type: SearchType = SearchType.RUSSIAN, n_links: int = 5) -> list:
        """
        Выполняет поисковый запрос и возвращает список ссылок.

        Args:
            query_text (str): Текст поискового запроса.
            search_type (SearchType, optional): Тип поиска. По умолчанию SearchType.RUSSIAN.
            n_links (int, optional): Количество ссылок для возврата. По умолчанию 5.

        Returns:
            list: Список найденных URL-адресов.
        """
        try:
            links = self.client.get_links(
                query_text=query_text,
                search_type=search_type,
                n_links=n_links
            )
            return links
        except Exception as e:
            logging.error(f"Произошла ошибка при выполнении поиска: {e}")
            return []
        

# --- Блок для тестового запуска ---
if __name__ == '__main__':
    """
    Этот блок кода выполняется только тогда, когда скрипт запускается напрямую.
    Он не будет выполняться при импорте класса YandexSearcher в другой модуль.
    """
    print("--- Запуск тестового режима для YandexSearcher ---")
    
    # Загружаем учетные данные из переменных окружения
    # Как получить folder_id: https://yandex.cloud/ru/docs/resource-manager/operations/folder/get-id
    # Как получить oauth_token: https://yandex.cloud/ru/docs/iam/concepts/authorization/oauth-token
    FOLDER_ID = os.getenv("FOLDER_ID")
    OAUTH_TOKEN = os.getenv("OAUTH_TOKEN")
    
    if not FOLDER_ID or not OAUTH_TOKEN:
        print("\nОшибка: Переменные окружения FOLDER_ID и OAUTH_TOKEN не установлены.")
        print("Пожалуйста, установите их перед запуском теста.")
    else:
        try:
            # 1. Создаем экземпляр класса
            searcher = YandexSearcher(folder_id=FOLDER_ID, oauth_token=OAUTH_TOKEN)
            
            # 2. Определяем тестовый запрос
            test_query = "У нас ИП на НДС. У нас с нового года в кассовых чеках при оплате наличными от юр.лиц пишется без НДС."
            
            # 3. Выполняем поиск
            results = searcher.search(query_text=test_query, n_links=3)
            
            # 4. Выводим результаты
            if results:
                print("\nРезультаты тестового поиска:")
                for index, link in enumerate(results, 1):
                    print(f"  {index}. {link}")
            else:
                print("\nПоиск не дал результатов или произошла ошибка (смотрите логи выше).")
                
        except ValueError as e:
            print(f"\nОшибка при создании экземпляра класса: {e}")
        except Exception as e:
            print(f"\nПроизошла непредвиденная ошибка во время теста: {e}")

    print("\n--- Тестовый режим завершен ---")