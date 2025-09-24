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

# --- Настройки по умолчанию для argparse ---
DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_QUERY = (
    "Как учесть расходы при передаче оборудования подрядчику для выполнения работ"
)
DEFAULT_SECTIONS = "law,recommendations"
DEFAULT_LIMIT = 15

# --- Настройки для Yandex Search API ---
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_OAUTH_TOKEN = os.getenv("YANDEX_OAUTH_TOKEN")


# Проверка наличия обязательных переменных окружения
if not all([ACTION_USERNAME, ACTION_PASSWORD, OPENAI_API_KEY, SERPER_API_KEY]):
    raise ValueError(
        "Необходимо задать все обязательные переменные окружения в файле .env: "
        "ACTION_USERNAME, ACTION_PASSWORD, OPENAI_API_KEY, SERPER_API_KEY"
    )