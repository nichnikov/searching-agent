import argparse
import config
from llm.llm_handler import LLMHandler
from searchers.internal_searcher import InternalSearcher
from searchers.google_searcher import WebSearcher
from prompts.templates import GENERATE_ANSWER_PROMPT_TEMPLATE, COMPARE_ANSWERS_PROMPT_TEMPLATE

class RAGPipeline:
    """
    Класс, инкапсулирующий логику RAG-пайплайна:
    поиск, генерация и сравнение ответов.
    """
    def __init__(self, llm_handler, internal_searcher, web_searcher):
        self.llm_handler = llm_handler
        self.internal_searcher = internal_searcher
        self.web_searcher = web_searcher

    def run(self, query: str, sections: list, limit: int):
        """Запускает полный цикл обработки запроса."""
        print(f"Обработка запроса: '{query}'")

        # --- Шаги 1-2: Поиск во внутренней базе и генерация ответа ---
        print("\n" + "="*20 + " ЭТАП 1: ВНУТРЕННИЙ ПОИСК " + "="*20)
        in_search_results = self.internal_searcher.search(query=query, sections=sections, limit=limit)
        in_answer = self._generate_answer(query, in_search_results, "внутренней базе")

        # --- Шаги 3-5: Поиск в интернете и генерация ответа ---
        print("\n" + "="*20 + " ЭТАП 2: ВНЕШНИЙ ПОИСК " + "="*20)
        out_search_results = self.web_searcher.search(query=query, num_results=limit)
        out_answer = self._generate_answer(query, out_search_results, "интернете")

        # --- Шаги 6-7: Сравнение ответов и генерация финального ---
        print("\n" + "="*20 + " ЭТАП 3: СРАВНЕНИЕ И ФИНАЛЬНЫЙ ОТВЕТ " + "="*20)
        final_answer = self._compare_and_finalize(query, in_answer, out_answer)

        # --- Вывод результатов ---
        self._print_results(in_answer, out_answer, final_answer)

    def _generate_answer(self, query: str, search_results: str, source_name: str) -> str:
        """Генерирует ответ LLM на основе результатов поиска."""
        print(f"Генерация ответа на основе поиска в {source_name}...")
        prompt = GENERATE_ANSWER_PROMPT_TEMPLATE.format(query=query, search_results=search_results)
        answer = self.llm_handler.get_response(prompt)
        print(f"✅ Ответ на основе {source_name} получен.")
        print(answer)
        return answer

    def _compare_and_finalize(self, query: str, in_answer: str, out_answer: str) -> str:
        """Сравнивает два ответа и генерирует итоговый."""
        print("Сравнение ответов и формирование итогового результата...")
        prompt = COMPARE_ANSWERS_PROMPT_TEMPLATE.format(query=query, in_answer=in_answer, out_answer=out_answer)
        final_answer = self.llm_handler.get_response(prompt)
        print("✅ Финальный ответ сформирован.")
        return final_answer

    def _print_results(self, in_answer: str, out_answer: str, final_answer: str):
        """Выводит все полученные ответы в консоль."""
        print("\n" + "#" * 50)
        print("### РЕЗУЛЬТАТЫ ###")
        print("#" * 50)

        print("\n" + "-"*15 + " Ответ на основе ВНУТРЕННИХ материалов " + "-"*15)
        print(in_answer)

        print("\n" + "-"*15 + " Ответ на основе ВНЕШНИХ материалов (Интернет) " + "-"*15)
        print(out_answer)

        print("\n" + "="*20 + " ИТОГОВЫЙ ОТВЕТ " + "="*20)
        print(final_answer)
        print("="*55)


def main():
    """Главная функция для запуска скрипта из командной строки."""
    parser = argparse.ArgumentParser(description="Запрос к LLM с предварительным поиском во внутренней и внешней базах.")
    parser.add_argument("--query", default=config.DEFAULT_QUERY, help="Текст запроса (вопрос)")
    parser.add_argument("--sections", default=config.DEFAULT_SECTIONS, help="Разделы для поиска через запятую")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="Количество статей для парсинга")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Имя модели LLM")
    args = parser.parse_args()

    try:
        # Инициализация компонентов
        llm = LLMHandler(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            model_name=args.model
        )
        internal_searcher = InternalSearcher(
            username=config.ACTION_USERNAME,
            password=config.ACTION_PASSWORD
        )
        web_searcher = WebSearcher()

        # Создание и запуск пайплайна
        pipeline = RAGPipeline(
            llm_handler=llm,
            internal_searcher=internal_searcher,
            web_searcher=web_searcher
        )
        pipeline.run(query=args.query, sections=args.sections.split(','), limit=args.limit)

    except (ValueError, ImportError) as e:
        print(f"\nОшибка инициализации: {e}")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка выполнения: {e}")

if __name__ == "__main__":
    main()