# main_rag_script.py

import argparse
import os
import json
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from scraper_runner import ActionScraperRunner # <-- НАШ НОВЫЙ ИМПОРТ

# Загружаем переменные из .env файла (рекомендуется)
load_dotenv()

# --- Настройки ---
# Используем переменные окружения для безопасности
# Создайте файл .env в корне проекта и добавьте в него:
# ACTION_USERNAME="ваш_логин"
# ACTION_PASSWORD="ваш_пароль"
ACTION_USERNAME = os.getenv("ACTION_USERNAME")
ACTION_PASSWORD = os.getenv("ACTION_PASSWORD")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_QUERY = "сотрудник идет в отпуск в январе 2025 расчетный период 2024 год Среднедневной заработок составил 680,48 нужно ли доводить отпускные до МРОТ 2025"

PROMPT_TEMPLATE = """
Твоя задача - помочь бухгалтеру, предоставив четкую рекомендацию на основе найденной во внутренней базе знаний информации.

Бухгалтерская ситуация: {query}

Был осуществлен поиск в корпоративной базе, вот найденные материалы:
---
{searching_results}
---

Твои действия:
1.  Внимательно изучи предоставленные материалы. Учитывай, что это самые релевантные внутренние документы.
2.  Выбери из них ключевые цитаты и выдержки, которые напрямую отвечают на вопрос бухгалтера.
3.  На основе этих цитат сформулируй ясную и однозначную рекомендацию со ссылками на законодательство, если оно упоминается в тексте.
4.  В конце ответа приведи ссылки на источники, которые ты использовал.
"""

def format_scraper_results(results: list[dict]) -> str:
    """Форматирует результаты от скрапера в читаемый текстовый блок для LLM."""
    if not results:
        return "Поиск во внутренней базе знаний не дал результатов."

    formatted_string = ""
    for i, item in enumerate(results, 1):
        title = item.get("title", "Без заголовка")
        url = item.get("url", "Ссылка отсутствует")
        content = item.get("content", "Содержимое отсутствует")
        
        # Обрезаем контент для экономии токенов, если он слишком длинный
        content_preview = (content[:2500] + '...') if len(content) > 2500 else content

        formatted_string += f"Источник #{i}:\n"
        formatted_string += f"  Название: {title}\n"
        formatted_string += f"  Ссылка: {url}\n"
        formatted_string += f"  Содержимое:\n\"\"\"\n{content_preview}\n\"\"\"\n\n"

    return formatted_string


def main() -> None:
    parser = argparse.ArgumentParser(description="Запрос к LLM с предварительным поиском во внутренней базе")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Текст запроса (вопрос)")
    parser.add_argument("--sections", default="law,recommendations", help="Разделы для поиска через запятую")
    parser.add_argument("--limit", type=int, default=3, help="Количество статей для парсинга")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Имя модели провайдера")
    args = parser.parse_args()

    # --- Шаг 1: Поиск информации с помощью скрапера ---
    print("=== Шаг 1: Выполняю поиск во внутренней базе... ===")
    try:
        runner = ActionScraperRunner(username=ACTION_USERNAME, password=ACTION_PASSWORD)
        scraped_items = runner.search(query=args.query, sections=args.sections.split(','), limit=args.limit)
    except (ValueError, ImportError) as e:
        print(f"Ошибка! Не удалось запустить скрапер: {e}")
        print("Убедитесь, что логин и пароль заданы в .env файле (ACTION_USERNAME, ACTION_PASSWORD).")
        return
    except Exception as e:
        print(f"Произошла непредвиденная ошибка во время работы скрапера: {e}")
        return

    # --- Шаг 2: Форматирование результатов и создание финального промпта ---
    print("\n=== Шаг 2: Форматирую результаты и готовлю промпт для LLM... ===")
    searching_results_text = format_scraper_results(scraped_items)
    final_prompt = PROMPT_TEMPLATE.format(query=args.query, searching_results=searching_results_text)
    
    # print(final_prompt) # Для отладки

    # --- Шаг 3: Обращение к LLM ---
    print(f"\n=== Шаг 3: Отправляю запрос к модели {args.model}... ===")
    try:
        llm = ChatOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, model=args.model)
        messages = [HumanMessage(content=final_prompt)]
        ai_msg = llm.invoke(messages)

        print("\n✅ === Ответ модели === ✅\n")
        print(ai_msg.content)
    except Exception as e:
        print(f"Произошла ошибка при обращении к LLM: {e}")


if __name__ == "__main__":
    main()