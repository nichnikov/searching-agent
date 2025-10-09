import os
import json
import argparse
from typing import List, Dict, Any, TypedDict, Set
from datetime import datetime
from abc import ABC, abstractmethod

import config  # Ваш файл конфигурации
from llm.llm_handler import LLMHandler
from searchers.combined_web_searcher import CombinedWebSearcher
from searchers.yandex_searcher import YandexSearcher
from utils.formatters import format_search_results # Может быть адаптирован, но для JSON-вывода реализован напрямую
from langgraph.graph import StateGraph, END
from prompts.templates import (
    SEARCH_QUERY_GENERATOR_PROMPT,
    # RESULTS_ANALYZER_PROMPT # Этот промт больше не используется
    FINAL_ANSWER_GENERATOR_PROMPT,
    PER_QUERY_ANALYZER_PROMPT
)

# --- Вспомогательные классы для поиска ---
# Для полноты примера, добавим BaseSearcher, который должен быть в .base_searcher

# --- Определение состояния графа (изменено) ---

class GraphState(TypedDict):
    original_query: str
    search_queries: List[str]
    # search_results: List[Dict[str, Any]] # Убрано, так как обработка идет по-другому
    # relevant_documents: List[Dict[str, Any]] # Убрано, так как анализ теперь встроен в поиск
    qa_results: List[Dict[str, Any]] # НОВОЕ: для хранения вопросов-ответов от LLM по каждому запросу
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
        # ИЗМЕНЕНИЕ: MultiQuerySearcher больше не нужен
        # self.multi_searcher = MultiQuerySearcher(self.web_searcher)
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Собирает граф LangGraph с узлами и переходами."""
        graph = StateGraph(GraphState)
        graph.add_node("generate_queries", self.generate_search_queries_node)
        graph.add_node("search_and_analyze_per_query", self.search_and_analyze_per_query_node) # НОВЫЙ УЗЕЛ
        graph.add_node("generate_answer", self.generate_final_answer_node)
        
        graph.set_entry_point("generate_queries")
        graph.add_edge("generate_queries", "search_and_analyze_per_query")
        
        # ИЗМЕНЕНИЕ: Условные переходы теперь зависят от нового узла
        graph.add_conditional_edges(
            "search_and_analyze_per_query",
            self.decide_next_step,
            {"RETRY": "generate_queries", "CONTINUE": "generate_answer", "END": END}
        )
        graph.add_edge("generate_answer", END)
        return graph.compile()

    # --- УЗЛЫ ГРАФА ---

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
        # rephrasing_count увеличиваем здесь, так как это цикл переформулирования запросов
        return {"search_queries": queries, "rephrasing_count": state['rephrasing_count'] + 1}

    def search_and_analyze_per_query_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Выполняет поиск по каждому сгенерированному запросу и немедленно анализирует результаты LLM,
        собирая ответы в список.
        """
        print("\n" + "="*20 + " ЭТАП 2: ПОИСК И АНАЛИЗ ПО КАЖДОМУ ЗАПРОСУ " + "="*20)

        original_query = state['original_query']
        search_queries = state['search_queries']
        # Используем существующие qa_results или создаем новый список
        all_qa_results = state.get('qa_results', [])

        current_batch_qa_results = [] # Для сбора результатов текущей итерации
        
        if not search_queries:
            print("Нет поисковых запросов для обработки.")
            return {"qa_results": all_qa_results, "feedback": "Нет запросов для поиска."}

        for i, single_query in enumerate(search_queries):
            print(f"Обработка поискового запроса [{i+1}/{len(search_queries)}]: '{single_query}'")
            
            # Выполняем поиск для одного запроса
            single_query_search_results = self.web_searcher.search(
                query=single_query,
                num_results=config.DEFAULT_LIMIT
            )

            if not single_query_search_results:
                print(f"Поиск для запроса '{single_query}' не дал результатов.")
                continue

            # Форматируем результаты поиска для промта LLM как JSON-строку массива объектов
            formatted_search_answer_blocks = []
            for doc in single_query_search_results:
                # Используем json.dumps для безопасного экранирования строковых значений
                # и затем обрезаем внешние кавычки, чтобы вставить их в f-строку с кавычками
                title = json.dumps(doc.get("title", ""))[1:-1]
                url = json.dumps(doc.get("url", ""))[1:-1]
                content = json.dumps(doc.get("content", ""))[1:-1]

                formatted_search_answer_blocks.append(
                    f'{{"title": "{title}", "url": "{url}", "content": "{content}"}}'
                )
            
            # Собираем в валидный JSON-массив
            formatted_search_answer_string = "[\n" + ",\n".join(formatted_search_answer_blocks) + "\n]"

            # Вызываем LLM для анализа каждого запроса
            prompt_for_analyzer = PER_QUERY_ANALYZER_PROMPT.format(
                query=original_query, # Отвечаем на оригинальный запрос пользователя
                search_answer=formatted_search_answer_string
            )
            response_from_llm = self.llm_handler.get_response(prompt_for_analyzer)

            try:
                # Удаляем возможные блоки Markdown "```json" и "```"
                cleaned_response = response_from_llm.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-len("```")].strip()

                llm_analysis = json.loads(cleaned_response)

                # Простая валидация структуры ответа LLM
                if all(k in llm_analysis for k in ["answer", "data"]) and \
                isinstance(llm_analysis["data"], dict) and \
                all(k in llm_analysis["data"] for k in ["urls", "title", "fragment"]):
                    
                    # Добавляем оригинальный поисковый запрос для контекста
                    llm_analysis['original_search_query_context'] = single_query
                    current_batch_qa_results.append(llm_analysis)
                    print(f"✅ Анализ для запроса '{single_query}' выполнен. Ответ получен.")
                else:
                    print(f"❌ LLM вернул некорректный JSON (неполная структура) для запроса '{single_query}'. Пропускаю.")
            except json.JSONDecodeError as e:
                print(f"❌ Не удалось распарсить JSON от LLM для запроса '{single_query}': {e}. Ответ LLM: '{response_from_llm}'. Пропускаю.")
            except Exception as e:
                print(f"❌ Непредвиденная ошибка при обработке ответа LLM для запроса '{single_query}': {e}. Пропускаю.")

        all_qa_results.extend(current_batch_qa_results)
        print(f"Собрано {len(current_batch_qa_results)} новых Q&A пар в текущем батче. Всего: {len(all_qa_results)}.")

        # Обратная связь для следующего шага
        feedback_message = ""
        if not current_batch_qa_results and len(all_qa_results) == 0:
            feedback_message = "Поиск и анализ не дали ни одной полезной Q&A пары."
        elif not current_batch_qa_results and len(all_qa_results) > 0:
            feedback_message = "Текущая партия запросов не дала новых полезных Q&A пар, но предыдущие попытки что-то нашли."
        else:
            feedback_message = "Получены новые Q&A пары."

        # Возвращаем накопленные qa_results и обновленную обратную связь
        return {"qa_results": all_qa_results, "feedback": feedback_message}

    # ИЗМЕНЕНИЕ: Узел analyze_results_node удален

    def generate_final_answer_node(self, state: GraphState) -> Dict[str, Any]:
        """Генерирует финальный ответ на основе собранных Q&A пар."""
        print("\n" + "="*20 + " ЭТАП 3: ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА " + "="*20)
        original_query = state['original_query']
        qa_results = state.get('qa_results', [])

        if not qa_results:
            print("Не найдено Q&A пар для генерации финального ответа.")
            return {"final_answer": "Не удалось сгенерировать ответ на основе найденной информации."}
        
        # Форматируем собранные Q&A пары для промта финального ответа
        formatted_qa_for_final_answer = ""
        for i, qa_item in enumerate(qa_results):
            # Проверяем, что ответ и данные не пустые или содержат значимую информацию
            # Это помогает отфильтровать бесполезные ответы LLM
            if qa_item.get("answer") and qa_item.get("answer").strip() and \
            qa_item.get("data", {}).get("urls") and qa_item.get("data", {}).get("urls").strip():
                answer = qa_item.get("answer", "Нет ответа.")
                data = qa_item.get("data", {})
                url = data.get("urls", "Нет ссылки.")
                title = data.get("title", "Нет заголовка.")
                # fragment = data.get("fragment", "Нет фрагмента.") # Может быть слишком много текста
                search_query_context = qa_item.get("original_search_query_context", "Неизвестный запрос.")

                formatted_qa_for_final_answer += f"--- Источник {i+1} (поисковый запрос: '{search_query_context}') ---\n"
                formatted_qa_for_final_answer += f"Заголовок: {title}\n"
                formatted_qa_for_final_answer += f"Ссылка: {url}\n"
                formatted_qa_for_final_answer += f"Краткий ответ по этому источнику: {answer}\n\n"

        if not formatted_qa_for_final_answer:
            print("Все Q&A пары оказались пустыми или нерелевантными.")
            return {"final_answer": "Не удалось сгенерировать ответ на основе найденной информации."}

        # Используем FINAL_ANSWER_GENERATOR_PROMPT, передавая ему отформатированные Q&A пары
        prompt = FINAL_ANSWER_GENERATOR_PROMPT.format(query=original_query, search_results=formatted_qa_for_final_answer)
        answer = self.llm_handler.get_response(prompt)
        print("✅ Финальный ответ сгенерирован.")
        return {"final_answer": answer}

    # --- УСЛОВНЫЕ ПЕРЕХОДЫ И ЗАПУСК ---
    def decide_next_step(self, state: GraphState) -> str:
        print("\n" + "="*20 + " ПРИНЯТИЕ РЕШЕНИЯ " + "="*20)
        # Решение теперь основывается на наличии собранных Q&A пар
        if state.get('qa_results') and len(state['qa_results']) > 0:
            # Проверяем, есть ли хоть одна действительно содержательная Q&A пара
            meaningful_qa_found = any(
                item.get("answer") and item["answer"].strip() and 
                item.get("data", {}).get("urls") and item["data"]["urls"].strip()
                for item in state['qa_results']
            )
            if meaningful_qa_found:
                print(f"✅ Найдено {len(state['qa_results'])} Q&A пар (из них содержательных: {sum(1 for item in state['qa_results'] if item.get('answer') and item['answer'].strip())}). Перехожу к генерации финального ответа.")
                return "CONTINUE"
        
        # Если Q&A пар не найдено после всех попыток переформулирования запросов
        if state['rephrasing_count'] >= self.max_retries:
            print("❌ Достигнут лимит попыток переформулирования запросов и поиска. Завершаю работу.")
            # Устанавливаем дефолтный финальный ответ, если ничего не удалось найти
            if not state.get('final_answer'): # Если ответ еще не установлен другим путем
                state['final_answer'] = "К сожалению, не удалось найти релевантную информацию или сгенерировать ответы после нескольких попыток."
            return "END"
        
        # Если Q&A пары не найдены, но есть еще попытки, возвращаемся к генерации новых запросов
        print(f"⚠️ Поиск и анализ не дали полезных Q&A пар. Попытка №{state['rephrasing_count'] + 1}. Возвращаюсь к переформулированию запросов.")
        return "RETRY"

    def run(self, query: str):
        initial_state = GraphState(
            original_query=query,
            search_queries=[],
            qa_results=[], # Инициализируем новый список
            feedback="",
            rephrasing_count=0,
            final_answer=""
        )
        print(f"Обработка запроса: '{query}'")
        final_state = self.graph.invoke(initial_state, config={"callbacks": [self.llm_handler.langfuse_handler]})
        # Дополнительная проверка на случай, если final_answer не был установлен в decide_next_step
        if not final_state.get('final_answer'):
            final_state['final_answer'] = "Не удалось сгенерировать ответ на основе имеющейся информации."
        self._print_final_result(final_state)
        self._save_results_to_json(final_state)
        return final_state

    def _print_final_result(self, final_state: GraphState):
            print("\n" + "#" * 60 + "\n###" + " " * 22 + "ИТОГИ ВЫПОЛНЕНИЯ" + " " * 22 + "###\n" + "#" * 60)
            print("\n" + "="*23 + " ФИНАЛЬНЫЙ ОТВЕТ " + "="*23)
            print(final_state['final_answer'])
            print("="*60)
        
    def _save_results_to_json(self, final_state: GraphState):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_log_{timestamp}.json"
        state_to_save = final_state.copy()
        try:
            os.makedirs("data", exist_ok=True) # Убедимся, что папка "data" существует
            with open(os.path.join("data", filename), 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4)
            print(f"\n[ИНФО] Полный лог выполнения сохранен в файл: {filename}")
        except Exception as e:
            print(f"\n[ОШИБКА] Не удалось сохранить лог в файл: {e}")


class ComponentFactory:
    
    def create_llm_handler(self, model_name: str) -> LLMHandler:
        return LLMHandler(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            model_name=model_name
        )

    def create_combined_web_searcher(self) -> CombinedWebSearcher:
        web_searcher_instances = []
        if hasattr(config, 'YANDEX_FOLDER_ID') and config.YANDEX_FOLDER_ID and hasattr(config, 'YANDEX_OAUTH_TOKEN') and config.YANDEX_OAUTH_TOKEN:
            web_searcher_instances.append(YandexSearcher(
            folder_id=config.YANDEX_FOLDER_ID,
            oauth_token=config.YANDEX_OAUTH_TOKEN
            ))
            print("Yandex Search провайдер активирован.")
        else:
            print("Yandex Search провайдер не сконфигурирован (отсутствуют YANDEX_FOLDER_ID или YANDEX_OAUTH_TOKEN).")