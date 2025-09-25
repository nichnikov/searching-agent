import argparse
from typing import List, Dict, Any

import config
from llm.llm_handler import LLMHandler
from prompts.templates import EDITOR_AGENT_PROMPT # COMPARE_ANSWERS_PROMPT_SECOND больше не нужен
from searchers.combined_web_searcher import CombinedWebSearcher
# Для ясности переименуем импорт, чтобы было понятно, что это поисковик Google
from searchers.google_searcher import WebSearcher as GoogleSearcher
from searchers.yandex_searcher import YandexSearcher
from utils.formatters import format_search_results


class WebSearchPipeline:
    """
    Класс, инкапсулирующий логику пайплайна:
    поиск в вебе (Google + Yandex) -> генерация финального ответа.
    """
    def __init__(self, llm_handler: LLMHandler, web_searcher: CombinedWebSearcher):
        self.llm_handler = llm_handler
        self.web_searcher = web_searcher

    def run(self, query: str, limit: int) -> str:
        """Запускает полный цикл обработки запроса."""
        print(f"Обработка запроса: '{query}'")

        # --- Этап 1: Поиск в Интернете (Google + Yandex) ---
        print("\n" + "="*20 + " ЭТАП 1: ПОИСК В ИНТЕРНЕТЕ " + "="*20)
        web_search_results = self.web_searcher.search(query=query, num_results=limit)

        # --- Этап 2: Генерация ответа на основе найденного ---
        print("\n" + "="*20 + " ЭТАП 2: ГЕНЕРАЦИЯ ОТВЕТА " + "="*20)
        final_answer = self._generate_answer_from_web(query, web_search_results)

        self._print_final_result(final_answer)
        
        return final_answer

    def _generate_answer_from_web(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Генерирует ответ LLM на основе результатов поиска в вебе."""
        source_name = "интернете"
        print(f"Форматирование {len(search_results)} результатов из '{source_name}' для LLM...")
        formatted_text = format_search_results(search_results, source_name)
        
        print(f"Генерация ответа на основе поиска в '{source_name}'...")
        prompt = EDITOR_AGENT_PROMPT.format(query=query, search_results=formatted_text)
        answer = self.llm_handler.get_response(prompt)
        print("✅ Ответ на основе веб-поиска получен.")
        return answer

    def _print_final_result(self, final_answer: str):
        """Выводит итоговый ответ в консоль."""
        print("\n" + "#" * 60)
        print("###" + " " * 22 + "ИТОГИ ВЫПОЛНЕНИЯ" + " " * 22 + "###")
        print("#" * 60)

        print("\n" + "="*23 + " ФИНАЛЬНЫЙ ОТВЕТ " + "="*23)
        print(final_answer)
        print("="*60)


class ComponentFactory:
    """
    Класс-фабрика, отвечающий за создание и конфигурацию
    всех необходимых компонентов пайплайна.
    """
    def create_llm_handler(self, model_name: str) -> LLMHandler:
        """Создает обработчик языковой модели."""
        return LLMHandler(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            model_name=model_name
        )

    def create_combined_web_searcher(self) -> CombinedWebSearcher:
        """
        Собирает и возвращает комбинированный поисковик,
        включая всех доступных внешних провайдеров.
        """
        web_searcher_instances = []
        if config.SERPER_API_KEY:
            web_searcher_instances.append(GoogleSearcher())
            print("Google Search провайдер активирован.")
        if config.YANDEX_FOLDER_ID and config.YANDEX_OAUTH_TOKEN:
            web_searcher_instances.append(YandexSearcher(
                folder_id=config.YANDEX_FOLDER_ID, 
                oauth_token=config.YANDEX_OAUTH_TOKEN
            ))
            print("Yandex Search провайдер активирован.")

        if not web_searcher_instances:
            raise ValueError("Не сконфигурирован ни один внешний поисковый провайдер (Google или Yandex).")

        return CombinedWebSearcher(searchers=web_searcher_instances)

    def create_web_search_pipeline(self, args: argparse.Namespace) -> WebSearchPipeline:
        """Создает и собирает готовый к работе пайплайн."""
        llm_handler = self.create_llm_handler(args.model)
        combined_searcher = self.create_combined_web_searcher()
        
        return WebSearchPipeline(
            llm_handler=llm_handler,
            web_searcher=combined_searcher
        )


def main():
    """Главная функция: парсинг аргументов и запуск пайплайна."""
    parser = argparse.ArgumentParser(description="Запрос к LLM с поиском в Google и Yandex.")
    parser.add_argument("--query", default=config.DEFAULT_QUERY, help="Текст запроса")
    # Аргумент --sections больше не нужен, т.к. нет внутреннего поиска
    # parser.add_argument("--sections", default=config.DEFAULT_SECTIONS, help="Разделы для внутреннего поиска")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="Количество результатов для каждого поисковика")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Имя модели LLM")
    args = parser.parse_args()

    try:
        # --- Сборка и запуск ---
        # 1. Создаем фабрику, которая знает, как собирать компоненты
        factory = ComponentFactory()
        # 2. Фабрика создает готовый пайплайн со всеми зависимостями
        pipeline = factory.create_web_search_pipeline(args)
        # 3. Запускаем процесс
        pipeline.run(query=args.query, limit=args.limit)

    except (ValueError, ImportError) as e:
        print(f"\n[ОШИБКА ИНИЦИАЛИЗАЦИИ] {e}")
    except Exception as e:
        print(f"\n[НЕПРЕДВИДЕННАЯ ОШИБКА] {e}")


if __name__ == "__main__":
    main()