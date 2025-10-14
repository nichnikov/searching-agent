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
затраты в черной металлургии от производства чугуна до выпуска проката.
- порядок бухгалтерского учета движения полуфабрикатов собственного производства между цехами (переделами),
- порядок учета затрат на производство и калькулирования себестоимости полуфабрикатов собственного производства,
- порядок отражения полуфабрикатов собственного производства в калькуляции при их производстве и дальнейшем использовании в производстве следующего передела.

особенности учета: 
- налогового, 
- бухгалтерского

другие важные моменты:
- нормативы расхода материалов на единицу продукции, если они есть
- бухгалтерские проводки
- влияние на себестоимость продукции (прямые и косвенные затраты)
- влияние на цену продукции
"""


DEFAULT_MODEL = "google/gemini-2.5-pro"
MODEL_CONTEXT_WINDOW = 1000000 
CONTENT_TOKEN_THRESHOLD = 10000 
MAX_TOKENS_FINAL_ANSWER = 25000
DEFAULT_QUERY = (query)
DEFAULT_SECTIONS = "law,recommendations"
DEFAULT_LIMIT = 5
MAX_RETRIES = 1

# --- Настройки для Yandex Search API ---
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_OAUTH_TOKEN = os.getenv("YANDEX_OAUTH_TOKEN")
DATA_DIR = "data"

# Проверка наличия обязательных переменных окружения
if not all([ACTION_USERNAME, ACTION_PASSWORD, OPENAI_API_KEY, SERPER_API_KEY]):
    raise ValueError(
        "Необходимо задать все обязательные переменные окружения в файле .env: "
        "ACTION_USERNAME, ACTION_PASSWORD, OPENAI_API_KEY, SERPER_API_KEY"
    )