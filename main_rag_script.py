# main_rag_script.py

import argparse
import os
import json
from dotenv import load_dotenv
from searchers.google_searcher import WebSearcher

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from searchers.scraper_runner import ActionScraperRunner # <-- НАШ НОВЫЙ ИМПОРТ

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
DEFAULT_QUERY = "Добрый день! Такой вопрос. У нас ИП на НДС. У нас с нового года в кассовых чеках при оплате наличными от юр.лиц пишется без НДС. Это правильно или так не должно быть."

PROMPT_TEMPLATE_1 = """
Твоя задача - помочь бухгалтеру, предоставив четкую рекомендацию на основе найденной во внутренней базе знаний информации.

Бухгалтерская ситуация: {query}

Был осуществлен поиск, вот найденные материалы:
---
{searching_results}
---

Твои действия:
1.  Внимательно изучи предоставленные материалы.
2.  Выбери из них ключевые цитаты и выдержки, которые напрямую отвечают на вопрос бухгалтера.
3.  На основе этих цитат сформулируй ясную и однозначную рекомендацию со ссылками на законодательство, если оно упоминается в тексте.
4.  В конце ответа приведи ссылки на источники, которые ты использовал.
"""

PROMPT_TEMPLATE_2 = """
Ниже два ответа с обоснованием на бухгалтерский вопрос (проблему) {query}


Ответ на базе поиска во внутренней базе знаний:
---
{in_answer}
---

Ответ на базе поиска в Интернете:
---
{out_answer}
---

Твои действия:
1. Внимательно изучи вопрос и предоставленные ответы.
2. Сравни ответ из внутренней базы и ответ из Интернета. Обращай особое внимание на ссылки на законодательство.
3. Если ответы совпадают, верни ответ из внутренней Базы знаний. 
4. Если ответы НЕ совпадают: Верни ответ из внешней базы, при этом верни наиболее подходящую ссылку из Внутренней Базы. И ссылки и названия материалов из Внешней Базы.
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
        content_preview = (content[:2500] + '...') if len(content) > 2500000 else content

        formatted_string += f"Источник #{i}:\n"
        formatted_string += f"  Название: {title}\n"
        formatted_string += f"  Ссылка: {url}\n"
        formatted_string += f"  Содержимое:\n\"\"\"\n{content_preview}\n\"\"\"\n\n"

    return formatted_string


def main() -> None:
    parser = argparse.ArgumentParser(description="Запрос к LLM с предварительным поиском во внутренней базе")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Текст запроса (вопрос)")
    parser.add_argument("--sections", default="law,recommendations", help="Разделы для поиска через запятую")
    parser.add_argument("--limit", type=int, default=15, help="Количество статей для парсинга")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Имя модели провайдера")
    args = parser.parse_args()

    # --- Шаг 1: Поиск информации с помощью скрапера ---
    print("=== Шаг 1: Выполняю поиск во внутренней базе... ===")
    try:
        runner = ActionScraperRunner(username=ACTION_USERNAME, password=ACTION_PASSWORD)
        in_results_text = runner.search(query=args.query, sections=args.sections.split(','), limit=args.limit)
        in_text_formated = format_scraper_results(in_results_text)
    except (ValueError, ImportError) as e:
        print(f"Ошибка! Не удалось запустить скрапер: {e}")
        print("Убедитесь, что логин и пароль заданы в .env файле (ACTION_USERNAME, ACTION_PASSWORD).")
        return
    except Exception as e:
        print(f"Произошла непредвиденная ошибка во время работы скрапера: {e}")
        return
    
    print("\n=== Форматирую результаты поиска по внутренним документам и готовлю промпт для LLM... ===")
    in_prompt = PROMPT_TEMPLATE_1.format(query=args.query, searching_results=in_text_formated)
    
    # --- Шаг 2: Обращение к LLM ---
    print(f"\n=== Шаг 2: Отправляю запрос к модели {args.model}... ===")
    try:
        llm = ChatOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, model=args.model)
        messages = [HumanMessage(content=in_prompt)]
        in_ai_msg = llm.invoke(messages)

        print("\n✅ === Ответ модели на базе своих материалов === ✅\n")
        in_search_answer = in_ai_msg.content
        print(in_search_answer)
    
    except Exception as e:
        print(f"Произошла ошибка при обращении к LLM: {e}")

    # --- Шаг 3: Поиск информации в интернете ---
    print("=== Шаг 3: Выполняю поиск в интернете... ===")
    try:
        # Убедитесь, что SERPER_API_KEY установлен как переменная окружения
        searcher = WebSearcher()
        out_search_results = searcher.search_internet(args.query, num_results=15)
        
        # Для отладки можно посмотреть результаты поиска
        # print(json.dumps(search_results, ensure_ascii=False, indent=2))

    except ValueError as e:
        print(f"Ошибка! Не удалось инициализировать WebSearcher: {e}")
        print("Пожалуйста, установите переменную окружения SERPER_API_KEY.")
        return
    except Exception as e:
        print(f"Произошла ошибка во время поиска: {e}")
        return


    # --- Шаг 4: Форматирование результатов и создание финального промпта ---
    print("\n=== Шаг 4: Форматирую результаты и готовлю промпт для LLM... ===")
    out_prompt = PROMPT_TEMPLATE_1.format(query=args.query, searching_results=out_search_results)
    
    # print(final_prompt) # Для отладки

    # --- Шаг 5: Обращение к LLM ---
    print(f"\n=== Шаг 3: Отправляю запрос к модели {args.model}... ===")
    try:
        llm = ChatOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, model=args.model)
        messages = [HumanMessage(content=out_prompt)]
        out_ai_msg = llm.invoke(messages)
        out_search_answer = out_ai_msg.content
        print("\n✅ === Ответ модели === ✅\n")
        print(out_search_answer)

    except Exception as e:
        print(f"Произошла ошибка при обращении к LLM: {e}")

    # --- Шаг 6: Сравнение результатов поиска по внутренней и внешней базам ---
    print("\n=== Шаг 6: Форматирую результаты и готовлю промпт для LLM... ===")
    compare_prompt = PROMPT_TEMPLATE_2.format(query=args.query, in_answer=in_search_answer, out_answer=out_search_answer)
    
    # print(final_prompt) # Для отладки

    # --- Шаг 5: Обращение к LLM ---
    print(f"\n=== Шаг 7: Отправляю запрос к модели {args.model}... ===")
    try:
        llm = ChatOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, model=args.model)
        messages = [HumanMessage(content=compare_prompt)]
        finaly_ai_msg = llm.invoke(messages)
        finaly_answer = finaly_ai_msg.content
        print("\n✅ === Конечный Ответ модели === ✅\n")
        print(finaly_answer)

    except Exception as e:
        print(f"Произошла ошибка при обращении к LLM: {e}")


    print(f"\nОтвет на внутренних материалах: " + "=" * 10)
    print(in_search_answer)

    print(f"\nОтвет на внешних материалах: " + "=" * 10)
    print(out_search_answer)

if __name__ == "__main__":
    main()