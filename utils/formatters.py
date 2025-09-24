from typing import List, Dict

def format_search_results(results: List[Dict], source_name: str) -> str:
    """
    Форматирует результаты от скрапера/поисковика в читаемый текстовый блок для LLM.
    """
    if not results:
        return f"Поиск в {source_name} не дал результатов."

    formatted_string = ""
    for i, item in enumerate(results, 1):
        title = item.get("title", "Без заголовка")
        url = item.get("url", "Ссылка отсутствует")
        content = item.get("content", "Содержимое отсутствует")

        # Обрезаем контент для экономии токенов
        content_preview = (content[:2500] + '...') if len(content) > 2500 else content

        formatted_string += f"Источник #{i}:\n"
        formatted_string += f"  Название: {title}\n"
        formatted_string += f"  Ссылка: {url}\n"
        formatted_string += f"  Содержимое:\n\"\"\"\n{content_preview}\n\"\"\"\n\n"

    return formatted_string