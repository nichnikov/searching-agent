import argparse
from typing import List, Dict, Any

import config
from llm.llm_processor import LLMHandler
from prompts.templates import GENERATE_ANSWER_PROMPT_FIRST, COMPARE_ANSWERS_PROMPT_SECOND
from searchers.combined_web_searcher import CombinedWebSearcher
from searchers.internal_searcher import InternalSearcher
# Для ясности переименуем импорт, чтобы было понятно, что это поисковик Google
from searchers.google_searcher import WebSearcher as GoogleSearcher
from searchers.yandex_searcher import YandexSearcher
from utils.formatters import format_search_results


class RAGPipeline:
    """
    Класс, инкапсулирующий полную логику RAG-пайплайна:
    поиск -> генерация -> сравнение -> финальный ответ.
    """
    def __init__(self, llm_handler: LLMHandler, internal_searcher: InternalSearcher, web_searcher: CombinedWebSearcher):
        self.llm_handler = llm_handler
        self.internal_searcher = internal_searcher
        self.web_searcher = web_searcher

    def run(self, query: str, sections: list, limit: int):
        """Запускает полный цикл обработки запроса."""
        print(f"Обработка запроса: '{query}'")

        # --- Этап 1: Внутренний поиск ---
        print("\n" + "="*20 + " ЭТАП 1: ВНУТРЕННИЙ ПОИСК " + "="*20)
        in_search_results = self.internal_searcher.search(query=query, sections=sections, limit=limit)
        in_answer = self._generate_answer(query, in_search_results, "внутренней базе")

        # --- Этап 2: Внешний поиск (через агрегатор) ---
        print("\n" + "="*20 + " ЭТАП 2: ВНЕШНИЙ ПОИСК (Google + Yandex) " + "="*20)
        out_search_results = self.web_searcher.search(query=query, num_results=limit)
        out_answer = self._generate_answer(query, out_search_results, "интернете")

        # --- Этап 3: Сравнение и финальный ответ ---
        print("\n" + "="*20 + " ЭТАП 3: СРАВНЕНИЕ И ФИНАЛЬНЫЙ ОТВЕТ " + "="*20)
        final_answer = self._compare_and_finalize(query, in_answer, out_answer)

        self._print_results(in_answer, out_answer, final_answer)

    def _generate_answer(self, query: str, search_results: List[Dict[str, Any]], source_name: str) -> str:
        """Генерирует ответ LLM, форматируя структурированные результаты поиска."""
        print(f"Форматирование {len(search_results)} результатов из '{source_name}' для LLM...")
        formatted_text = format_search_results(search_results, source_name)
        
        print(f"Генерация ответа на основе поиска в '{source_name}'...")
        prompt = GENERATE_ANSWER_PROMPT_FIRST.format(query=query, search_results=formatted_text)
        answer = self.llm_handler.get_response(prompt)
        print(f"✅ Ответ на основе '{source_name}' получен.")
        return answer

    def _compare_and_finalize(self, query: str, in_answer: str, out_answer: str) -> str:
        """Сравнивает два ответа и генерирует итоговый."""
        print("Сравнение ответов и формирование итогового результата...")
        prompt = COMPARE_ANSWERS_PROMPT_SECOND.format(
            query=query, 
            in_answer=in_answer, 
            out_answer=out_answer
        )
        final_answer = self.llm_handler.get_response(prompt)
        print("✅ Финальный ответ сформирован.")
        return final_answer

    def _print_results(self, in_answer: str, out_answer: str, final_answer: str):
        """Выводит все полученные ответы в консоль для наглядного сравнения."""
        print("\n" + "#" * 60)
        print("###" + " " * 22 + "ИТОГИ ВЫПОЛНЕНИЯ" + " " * 22 + "###")
        print("#" * 60)

        print("\n" + "-"*15 + " Ответ на основе ВНУТРЕННИХ материалов " + "-"*15)
        print(in_answer)

        print("\n" + "-"*15 + " Ответ на основе ВНЕШНИХ материалов (Интернет) " + "-"*14)
        print(out_answer)

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

    def create_internal_searcher(self) -> InternalSearcher:
        """Создает поисковик по внутренней базе."""
        return InternalSearcher(
            username=config.ACTION_USERNAME,
            password=config.ACTION_PASSWORD
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

    def create_rag_pipeline(self, args: argparse.Namespace) -> RAGPipeline:
        """Создает и собирает готовый к работе RAG-пайплайн."""
        llm_handler = self.create_llm_handler(args.model)
        internal_searcher = self.create_internal_searcher()
        combined_searcher = self.create_combined_web_searcher()
        
        return RAGPipeline(
            llm_handler=llm_handler,
            internal_searcher=internal_searcher,
            web_searcher=combined_searcher
        )


def main():
    """Главная функция: парсинг аргументов и запуск пайплайна."""
    parser = argparse.ArgumentParser(description="Запрос к LLM с комбинированным поиском.")
    parser.add_argument("--query", default=config.DEFAULT_QUERY, help="Текст запроса")
    parser.add_argument("--sections", default=config.DEFAULT_SECTIONS, help="Разделы для внутреннего поиска")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="Количество результатов для каждого поисковика")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Имя модели LLM")
    args = parser.parse_args()

    try:
        # --- Сборка и запуск ---
        # 1. Создаем фабрику, которая знает, как собирать компоненты
        factory = ComponentFactory()
        # 2. Фабрика создает готовый пайплайн со всеми зависимостями
        pipeline = factory.create_rag_pipeline(args)
        # 3. Запускаем процесс
        pipeline.run(query=args.query, sections=args.sections.split(','), limit=args.limit)

    except (ValueError, ImportError) as e:
        print(f"\n[ОШИБКА ИНИЦИАЛИЗАЦИИ] {e}")
    except Exception as e:
        print(f"\n[НЕПРЕДВИДЕННАЯ ОШИБКА] {e}")


if __name__ == "__main__":
    main()