import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Настройки API и моделей ---
ACTION_USERNAME = os.getenv("ACTION_USERNAME")
ACTION_PASSWORD = os.getenv("ACTION_PASSWORD")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY") # Добавляем ключ для Serper
LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST=os.getenv("LANGFUSE_HOST")

# --- Настройки по умолчанию для argparse ---

query = """
ФСБУ 9/2025 ввели в действие в середине года. 
Приведи обоснование, что компания может его применять с 1 января 2025 года, то есть задним числом. 
В каком периоде делать проводки по переходу: на дату принятия ФСБУ или на последнее число года? 
Как корректировать учет с начала 2025 года до перехода? Через какие счета: 90 или 84? 
Если компания сдает ежеквартальную промежуточную отчетность и она приняла решение перейти на ФСБ 9/2025 досрочно с 1 января 2025 года, обязана ли она отчетность за 9 месяцев сдать уже по новому стандарту? 
или она вправе сдать отчетность по ПБУ 9/99, а за год переделать учет по новому стандарту?
"""

DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_QUERY = (query)
DEFAULT_SECTIONS = "law,recommendations"
DEFAULT_LIMIT = 5
MAX_RETRIES = 1

# --- Настройки для Yandex Search API ---
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_OAUTH_TOKEN = os.getenv("YANDEX_OAUTH_TOKEN")


# Проверка наличия обязательных переменных окружения
if not all([ACTION_USERNAME, ACTION_PASSWORD, OPENAI_API_KEY, SERPER_API_KEY]):
    raise ValueError(
        "Необходимо задать все обязательные переменные окружения в файле .env: "
        "ACTION_USERNAME, ACTION_PASSWORD, OPENAI_API_KEY, SERPER_API_KEY"
    )