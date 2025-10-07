import argparse
import os
import json

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from searchers.google_searcher import WebSearcher

# --- Настройки ---
# Рекомендуется вынести ключи в переменные окружения для безопасности
# os.environ["VSEGPT_API_KEY"] = "sk-or-vv-..."
# os.environ["SERPER_API_KEY"] = "..."

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = "google/gemini-2.5-pro"
# DEFAULT_QUERY = "сотрудник идет в отпуск в январе 2025 расчетный период 2024 год Среднедневной заработок составил 680,48 нужно ли доводить отпускные до МРОТ 2025"
DEFAULT_QUERY = "На предприятии дополнительные неоплачиваемые дни отдыха (отгул) за работу в выходной день оплачиваются путем уменьшения нормы рабочего времени. Данное действие утвердили положением об оплате труда с 1 марта. Положена ли такая схема при работе в выходной день до 1 марта?"
# Шаблон промпта, который будет заполнен результатами поиска
PROMPT_TEMPLATE = """
Твоя задача - помочь бухгалтеру, предоставив четкую рекомендацию на основе найденной в интернете информации.

Бухгалтерская ситуация: {query}

Был осуществлен поиск в Интернете, вот найденные материалы:
---
{searching_results}
---

Твои действия:
1.  Внимательно изучи предоставленные материалы.
2.  Выбери из них ключевые фразы и цитаты, которые напрямую отвечают на вопрос бухгалтера.
3.  На основе этих цитат сформулируй ясную и однозначную рекомендацию.
4.  В конце ответа приведи ссылки на источники, которые ты использовал.
"""

def format_search_results(results: dict) -> str:
    """Форматирует результаты поиска в читаемый текстовый блок."""
    if "error" in results or not results.get("organic"):
        return "Поиск в интернете не дал результатов."

    formatted_string = ""
    for i, item in enumerate(results["organic"], 1):
        title = item.get("title", "Без заголовка")
        link = item.get("link", "Ссылка отсутствует")
        snippet = item.get("snippet", "Описание отсутствует").replace("\n", " ")

        formatted_string += f"Источник #{i}:\n"
        formatted_string += f"  Название: {title}\n"
        formatted_string += f"  Ссылка: {link}\n"
        formatted_string += f"  Фрагмент: {snippet}\n\n"

    return formatted_string

def main() -> None:
    parser = argparse.ArgumentParser(description="Запрос к LLM с предварительным поиском в интернете")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Текст запроса (вопрос)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Имя модели провайдера")
    parser.add_argument("--temperature", type=float, default=0.2, help="Креативность ответа")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Лимит токенов ответа")
    parser.add_argument("--timeout", type=int, default=120, help="Таймаут запроса в секундах")
    args = parser.parse_args()

    # --- Шаг 1: Поиск информации в интернете ---
    print("=== Шаг 1: Выполняю поиск в интернете... ===")
    try:
        # Убедитесь, что SERPER_API_KEY установлен как переменная окружения
        searcher = WebSearcher()
        search_results = searcher.search_internet(args.query, num_results=5)
        
        # Для отладки можно посмотреть результаты поиска
        # print(json.dumps(search_results, ensure_ascii=False, indent=2))

    except ValueError as e:
        print(f"Ошибка! Не удалось инициализировать WebSearcher: {e}")
        print("Пожалуйста, установите переменную окружения SERPER_API_KEY.")
        return
    except Exception as e:
        print(f"Произошла ошибка во время поиска: {e}")
        return
        
    # --- Шаг 2: Форматирование результатов и создание финального промпта ---
    print("=== Шаг 2: Форматирую результаты и готовлю промпт для LLM... ===")
    searching_results_text = format_search_results(search_results)
    final_prompt = PROMPT_TEMPLATE.format(query=args.query, searching_results=searching_results_text)
    
    # Для отладки можно посмотреть финальный промпт
    # print("--- Финальный промпт ---")
    # print(final_prompt)
    # print("-----------------------")

    # --- Шаг 3: Обращение к LLM ---
    print(f"=== Шаг 3: Отправляю запрос к модели {args.model}... ===")
    try:
        llm = ChatOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

        messages = [HumanMessage(content=final_prompt)]
        ai_msg = llm.invoke(messages)

        print("\n✅ === Ответ модели === ✅\n")
        print(ai_msg.content)

        meta = getattr(ai_msg, "response_metadata", None) or {}
        usage = meta.get("usage", {}) # В некоторых API usage находится внутри response_metadata
        if meta or usage:
            print("\n--- Метаданные ---\n")
            if 'model_name' in meta: print(f"model: {meta['model_name']}")
            if 'token_usage' in meta:
                usage = meta['token_usage']
                print(f"prompt_tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"completion_tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"total_tokens: {usage.get('total_tokens', 'N/A')}")

    except Exception as e:
        print(f"Произошла ошибка при обращении к LLM: {e}")


if __name__ == "__main__":
    main()
    