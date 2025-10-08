import os
import json
import argparse

from typing import List, Dict, Any, TypedDict, Annotated
from datetime import datetime
import operator

import config  # Ваш файл конфигурации
from llm.llm_handler import LLMHandler
# Проигнорируем EDITOR_AGENT_PROMPT, так как у нас будут более гранулярные промпты
# from prompts.templates import EDITOR_AGENT_PROMPT
from searchers.combined_web_searcher import CombinedWebSearcher
from searchers.yandex_searcher import YandexSearcher
from utils.formatters import format_search_results
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langfuse import observe
# Убедитесь, что langgraph установлен: pip install langgraph
from langgraph.graph import StateGraph, END
from prompts.templates import (
    SEARCH_QUERY_GENERATOR_PROMPT,
    RESULTS_ANALYZER_PROMPT,
    FINAL_ANSWER_GENERATOR_PROMPT
    )



# --- Определение состояния графа ---

langfuse = get_client()
langfuse_handler = CallbackHandler()

class GraphState(TypedDict):
    """
    Определяет состояние нашего графа.
    
    Attributes:
        original_query: Исходный запрос пользователя.
        search_queries: Список поисковых запросов, сгенерированных LLM.
        search_results: Список результатов, полученных из поисковиков.
        relevant_documents: Отфильтрованные релевантные документы.
        feedback: Обратная связь от анализатора для улучшения поиска.
        rephrasing_count: Счетчик итераций переформулирования запроса.
        final_answer: Итоговый ответ для пользователя.
    """
    original_query: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    feedback: str
    rephrasing_count: int
    final_answer: str


class LangGraphPipeline:
    """
    Класс, инкапсулирующий логику пайплайна на основе LangGraph.
    """
    def __init__(self, llm_handler: LLMHandler, web_searcher: CombinedWebSearcher, max_retries: int = 2):
        self.llm_handler = llm_handler
        self.web_searcher = web_searcher
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Собирает граф LangGraph с узлами и переходами."""
        graph = StateGraph(GraphState)

        # Определение узлов графа
        graph.add_node("generate_queries", self.generate_search_queries_node)
        graph.add_node("web_search", self.web_search_node)
        graph.add_node("analyze_results", self.analyze_results_node)
        graph.add_node("generate_answer", self.generate_final_answer_node)

        # Определение переходов
        graph.set_entry_point("generate_queries")
        graph.add_edge("generate_queries", "web_search")
        graph.add_edge("web_search", "analyze_results")
        graph.add_conditional_edges(
            "analyze_results",
            self.decide_next_step,
            {
                "RETRY": "generate_queries",
                "CONTINUE": "generate_answer",
                "END": END
            }
        )
        graph.add_edge("generate_answer", END)

        return graph.compile()

    # --- УЗЛЫ ГРАФА ---

    @observe
    def generate_search_queries_node(self, state: GraphState) -> Dict[str, Any]:
        """Генерирует поисковые запросы на основе запроса пользователя и обратной связи."""
        print("\n" + "="*20 + " ЭТАП 1: ГЕНЕРАЦИЯ ПОИСКОВЫХ ЗАПРОСОВ " + "="*20)
        
        query = state['original_query']
        feedback = state.get('feedback', '')
        feedback_prompt = f"Учти предыдущую обратную связь: {feedback}" if feedback else ""

        prompt = SEARCH_QUERY_GENERATOR_PROMPT.format(query=query, feedback=feedback_prompt)
        response = self.llm_handler.get_response(prompt)
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        print(f"Сгенерированные запросы: {queries}")
        return {"search_queries": queries, "rephrasing_count": state['rephrasing_count'] + 1}

    @observe
    def web_search_node(self, state: GraphState) -> Dict[str, Any]:
        """Выполняет поиск в вебе по сгенерированным запросам."""
        print("\n" + "="*20 + " ЭТАП 2: ПОИСК В ИНТЕРНЕТЕ " + "="*20)
        
        all_results = []
        queries = state['search_queries']
        # Сохраняем все предыдущие результаты, чтобы не потерять их при повторном поиске
        previous_results = state.get('search_results', [])

        for query in queries:
            print(f"Поиск по запросу: '{query}'")
            # Предполагаем, что ваш searcher может принимать один запрос за раз
            results = self.web_searcher.search(query=query, num_results=config.DEFAULT_LIMIT)
            all_results.extend(results)
        
        print(f"Найдено {len(all_results)} результатов.")
        # Добавляем новые результаты к уже существующим
        return {"search_results": previous_results + all_results}

    @observe
    def analyze_results_node(self, state: GraphState) -> Dict[str, Any]:
        """Анализирует результаты поиска на релевантность."""
        print("\n" + "="*20 + " ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ " + "="*20)
        
        query = state['original_query']
        # Используем только последние результаты для анализа
        results_to_analyze = state['search_results'][len(state.get('relevant_documents', [])):]
        
        if not results_to_analyze:
             print("Новых результатов для анализа нет. Завершаю...")
             return {"relevant_documents": [], "feedback": "Поиск не дал новых результатов."}

        formatted_results = format_search_results(results_to_analyze, "веб-поиск")
        prompt = RESULTS_ANALYZER_PROMPT.format(query=query, search_results=formatted_results)
        
        response = self.llm_handler.get_response(prompt)
        
        try:
            # Извлекаем JSON из ответа LLM
            analysis = json.loads(response.strip().replace("```json", "").replace("```", ""))
            relevant_ids = analysis.get("relevant_ids", [])
            feedback = analysis.get("feedback", "")
            
            print(f"Анализ завершен. Релевантные ID: {relevant_ids}. Обратная связь: '{feedback}'")
            
            # Фильтруем документы
            relevant_docs = [doc for i, doc in enumerate(results_to_analyze) if i in relevant_ids]
            
            # Добавляем новые релевантные документы к уже существующим
            all_relevant_docs = state.get('relevant_documents', []) + relevant_docs
            return {"relevant_documents": all_relevant_docs, "feedback": feedback}

        except json.JSONDecodeError:
            print("[ОШИБКА] Не удалось распарсить JSON от LLM-анализатора.")
            return {"relevant_documents": [], "feedback": "Ошибка анализа результатов."}

    @observe
    def generate_final_answer_node(self, state: GraphState) -> Dict[str, Any]:
        """Генерирует финальный ответ на основе релевантных документов."""
        print("\n" + "="*20 + " ЭТАП 4: ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА " + "="*20)

        query = state['original_query']
        relevant_docs = state['relevant_documents']
        formatted_docs = format_search_results(relevant_docs, "проверенные источники")

        prompt = FINAL_ANSWER_GENERATOR_PROMPT.format(query=query, search_results=formatted_docs)
        answer = self.llm_handler.get_response(prompt)
        
        print("✅ Финальный ответ сгенерирован.")
        return {"final_answer": answer}

    # --- УСЛОВНЫЕ ПЕРЕХОДЫ ---
    @observe
    def decide_next_step(self, state: GraphState) -> str:
        """Решает, какой узел будет следующим."""
        print("\n" + "="*20 + " ПРИНЯТИЕ РЕШЕНИЯ " + "="*20)
        
        if state['relevant_documents']:
            print("✅ Результаты релевантны. Перехожу к генерации ответа.")
            return "CONTINUE"
        
        if state['rephrasing_count'] >= self.max_retries:
            print("❌ Достигнут лимит попыток. Завершаю работу.")
            final_answer = "К сожалению, не удалось найти релевантную информацию после нескольких попыток."
            state['final_answer'] = final_answer
            return "END"
        
        print(f"⚠️ Результаты нерелевантны. Попытка №{state['rephrasing_count'] + 1}. Возвращаюсь к переформулированию запросов.")
        return "RETRY"

    def run(self, query: str):
        """
        Запускает полный цикл обработки запроса.
        """
        initial_state = GraphState(
            original_query=query,
            search_queries=[],
            search_results=[],
            relevant_documents=[],
            feedback="",
            rephrasing_count=0,
            final_answer=""
        )
        
        print(f"Обработка запроса: '{query}'")
        
        # ИСПОЛЬЗУЕМ .invoke() ДЛЯ ПОЛУЧЕНИЯ ПОЛНОГО ФИНАЛЬНОГО СОСТОЯНИЯ
        # Все print() внутри узлов будут выведены в консоль в процессе выполнения.
        final_state = self.graph.invoke(initial_state, config={"callbacks": [langfuse_handler]})

        # Если в конце работы граф остановился из-за ошибки или лимита попыток,
        # а финальный ответ не был сгенерирован, установим заглушку.
        if not final_state.get('final_answer'):
            final_state['final_answer'] = "Не удалось сгенерировать ответ на основе имеющейся информации."

        self._print_final_result(final_state)
        self._save_results_to_json(final_state)
        
        return final_state
       

    def _print_final_result(self, final_state: GraphState):
            """Выводит итоговый ответ в консоль."""
            print("\n" + "#" * 60)
            print("###" + " " * 22 + "ИТОГИ ВЫПОЛНЕНИЯ" + " " * 22 + "###")
            print("#" * 60)

            print("\n" + "="*23 + " ФИНАЛЬНЫЙ ОТВЕТ " + "="*23)
            print(final_state['final_answer'])
            print("="*60)
        
    def _save_results_to_json(self, final_state: GraphState):
        """Сохраняет все запросы и результаты в JSON файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_log_{timestamp}.json"
        
        # Теперь final_state содержит все поля из GraphState
        state_to_save = final_state.copy()

        try:
            with open(os.path.join("data", filename), 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4)
            print(f"\n[ИНФО] Полный лог выполнения сохранен в файл: {filename}")
        except Exception as e:
            print(f"\n[ОШИБКА] Не удалось сохранить лог в файл: {e}")

# --- Фабрика и Main (остаются почти без изменений) ---

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
        """Собирает и возвращает комбинированный поисковик."""
        web_searcher_instances = []
        # Google Searcher можно добавить аналогично
        if config.YANDEX_FOLDER_ID and config.YANDEX_OAUTH_TOKEN:
            web_searcher_instances.append(YandexSearcher(
                folder_id=config.YANDEX_FOLDER_ID,
                oauth_token=config.YANDEX_OAUTH_TOKEN
            ))
            print("Yandex Search провайдер активирован.")

        if not web_searcher_instances:
            raise ValueError("Не сконфигурирован ни один внешний поисковый провайдер.")

        return CombinedWebSearcher(searchers=web_searcher_instances)

    def create_langgraph_pipeline(self, args: argparse.Namespace) -> LangGraphPipeline:
        """Создает и собирает готовый к работе пайплайн на LangGraph."""
        llm_handler = self.create_llm_handler(args.model)
        combined_searcher = self.create_combined_web_searcher()
        
        return LangGraphPipeline(
            llm_handler=llm_handler,
            web_searcher=combined_searcher
        )


def main():
    """Главная функция: парсинг аргументов и запуск пайплайна."""
    parser = argparse.ArgumentParser(description="Запрос к LLM с итеративным поиском в вебе через LangGraph.")
    parser.add_argument("--query", default=config.DEFAULT_QUERY, help="Текст запроса")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="Количество результатов для каждого поисковика")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Имя модели LLM")
    args = parser.parse_args()

    # Обновляем конфиг из аргументов, если это необходимо
    config.DEFAULT_LIMIT = args.limit

    try:
        factory = ComponentFactory()
        pipeline = factory.create_langgraph_pipeline(args)
        pipeline.run(query=args.query)

    except (ValueError, ImportError) as e:
        print(f"\n[ОШИБКА ИНИЦИАЛИЗАЦИИ] {e}")
    except Exception as e:
        print(f"\n[НЕПРЕДВИДЕННАЯ ОШИБКА] {e}")


if __name__ == "__main__":
    main()