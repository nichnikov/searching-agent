# main.py
import os
import json
import argparse
from typing import List, Dict, Any, TypedDict, Optional, Set
from datetime import datetime

# Импортируем модули из нашей структуры проекта
import config
from llm.llm_handler import LLMHandler
from searchers.base_searcher import BaseSearcher 
from searchers.combined_web_searcher import CombinedWebSearcher
from searchers.yandex_searcher import YandexSearcher
from utils.formatters import format_search_results # Сохранено для потенциального использования или отладки
from utils.str2dir import structure_text_to_json_list
from langgraph.graph import StateGraph, END
from prompts.templates import (
    SEARCH_QUERY_GENERATOR_PROMPT,
    PER_QUERY_ANALYZER_PROMPT, 
    FINAL_ANSWER_GENERATOR_PROMPT
)

# --- Определение состояния графа ---
class GraphState(TypedDict):
    original_query: str
    search_queries: List[str]
    qa_results: List[Dict[str, Any]] # Для хранения вопросов-ответов от LLM по каждому запросу
    feedback: str
    rephrasing_count: int
    final_answer: str

class LangGraphPipeline:
    """
    Класс, инкапсулирующий логику пайплайна на основе LangGraph.
    """
    def __init__(self, llm_handler: LLMHandler, web_searcher: CombinedWebSearcher, max_retries: int = config.MAX_RETRIES):
        self.llm_handler = llm_handler
        self.web_searcher = web_searcher
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Собирает граф LangGraph с узлами и переходами."""
        graph = StateGraph(GraphState)
        graph.add_node("generate_queries", self.generate_search_queries_node)
        graph.add_node("search_and_analyze_per_query", self.search_and_analyze_per_query_node)
        # graph.add_node("generate_answer", self.generate_final_answer_node)
        
        '''
        graph.set_entry_point("generate_queries")
        graph.add_edge("generate_queries", "search_and_analyze_per_query")
        
        graph.add_conditional_edges(
            "search_and_analyze_per_query",
            self.decide_next_step,
            {"RETRY": "generate_queries", "CONTINUE": "generate_answer", "END": END}
        )'''
        graph.add_node("generate_answer", self.generate_final_answer_node)
        # graph.add_edge("search_and_analyze_per_query", "generate_answer")
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
        all_qa_results = list(state.get('qa_results', [])) # Делаем изменяемую копию

        current_batch_qa_results = []
        
        if not search_queries:
            print("Нет поисковых запросов для обработки.")
            return {"qa_results": all_qa_results, "feedback": "Нет запросов для поиска."}

        for i, single_query in enumerate(search_queries):
            print(f"Обработка поискового запроса [{i+1}/{len(search_queries)}]: '{single_query}'")
            
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
                formatted_search_answer_blocks.append({
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "content": doc.get("content", "")
                })
            
            # Преобразуем список словарей в JSON-строку
            formatted_search_answer_string = json.dumps(formatted_search_answer_blocks, ensure_ascii=False, indent=2)

            # Вызываем LLM для анализа каждого запроса
            prompt_for_analyzer = PER_QUERY_ANALYZER_PROMPT.format(
                query=original_query,
                search_answer=formatted_search_answer_string
            )
            response_from_llm = self.llm_handler.get_response(
                prompt_for_analyzer,
                response_format={"type": "json_object"} # Запрашиваем JSON напрямую от LLM
            )

            try:
                llm_analysis_list = structure_text_to_json_list(response_from_llm)
                # response_from_llm_cleared = structure_text_to_json_list(response_from_llm)
                # llm_analysis = json.loads(response_from_llm_cleared)

                for llm_analysis_dict in llm_analysis_list:
                    # Простая валидация структуры ответа LLM
                    if all(k in llm_analysis_dict for k in ["answer", "data"]) and isinstance(llm_analysis_dict["data"], list):
                        # Добавляем оригинальный поисковый запрос для контекста
                        llm_analysis_dict['original_search_query_context'] = single_query
                        current_batch_qa_results.append(llm_analysis_dict)
                        print(f"✅ Анализ для запроса '{single_query}' выполнен. Ответ получен.")
                    else:
                        print(f"❌ LLM вернул некорректный JSON (неполная структура) для запроса '{single_query}'. Пропускаю. Ответ: {llm_analysis_list}")

            except json.JSONDecodeError as e:
                print(f"❌ Не удалось распарсить JSON от LLM для запроса '{single_query}': {e}. Ответ LLM: '{response_from_llm}'. Пропускаю.")

            except Exception as e:
                print(f"❌ Непредвиденная ошибка при обработке ответа LLM для запроса '{single_query}': {e}. Пропускаю.")

        all_qa_results.extend(current_batch_qa_results)
        print(f"Собрано {len(current_batch_qa_results)} новых Q&A пар в текущем батче. Всего: {len(all_qa_results)}.")

        feedback_message = ""

        if not current_batch_qa_results and len(all_qa_results) == 0:
            feedback_message = "Поиск и анализ не дали ни одной полезной Q&A пары."
        elif not current_batch_qa_results and len(all_qa_results) > 0:
            feedback_message = "Текущая партия запросов не дала новых полезных Q&A пар, но предыдущие попытки что-то нашли."
        else: 
            feedback_message = "Получены новые Q&A пары."

        return {"qa_results": all_qa_results, "feedback": feedback_message}

    def generate_final_answer_node(self, state: GraphState) -> Dict[str, Any]:
        """Генерирует финальный ответ на основе собранных Q&A пар."""
        print("\n" + "="*20 + " ЭТАП 3: ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА " + "="*20)
        original_query = state['original_query']
        qa_results = state.get('qa_results', [])

        if not qa_results:
            print("Не найдено Q&A пар для генерации финального ответа.")
            return {"final_answer": "Не удалось сгенерировать ответ на основе найденной информации."}
        
        formatted_qa_for_final_answer = ""
        for i, qa_item in enumerate(qa_results):
            # Проверяем, что ответ и данные не пустые или содержат значимую информацию
            if qa_item.get("answer") and qa_item.get("answer").strip() and \
               qa_item.get("data", {}).get("urls") and qa_item.get("data", {}).get("urls").strip():
                answer = qa_item.get("answer", "Нет ответа.")
                data = qa_item.get("data", {})
                url = data.get("urls", "Нет ссылки.")
                title = data.get("title", "Нет заголовка.")
                search_query_context = qa_item.get("original_search_query_context", "Неизвестный запрос.")

                formatted_qa_for_final_answer += f"--- Источник {i+1} (поисковый запрос: '{search_query_context}') ---\n"
                formatted_qa_for_final_answer += f"Заголовок: {title}\n"
                formatted_qa_for_final_answer += f"Ссылка: {url}\n"
                formatted_qa_for_final_answer += f"Краткий ответ по этому источнику: {answer}\n\n"

        if not formatted_qa_for_final_answer:
            print("Все Q&A пары оказались пустыми или нерелевантными.")
            return {"final_answer": "Не удалось сгенерировать ответ на основе найденной информации."}

        prompt = FINAL_ANSWER_GENERATOR_PROMPT.format(query=original_query, search_results=formatted_qa_for_final_answer)
        answer = self.llm_handler.get_response(prompt)
        print("✅ Финальный ответ сгенерирован.")
        return {"final_answer": answer}

    def decide_next_step(self, state: GraphState) -> str:
        """
        Определяет следующий шаг в пайплайне на основе результатов анализа.
        """
        print("\n" + "="*20 + " ПРИНЯТИЕ РЕШЕНИЯ " + "="*20)
        meaningful_qa_found_overall = any(
            item.get("answer") and item["answer"].strip() and 
            item.get("data", {}).get("urls") and item["data"]["urls"].strip()
            for item in state.get('qa_results', [])
        )
        
        if meaningful_qa_found_overall:
            print(f"✅ Найдено {len(state.get('qa_results', []))} Q&A пар (из них содержательных: {sum(1 for item in state.get('qa_results', []) if item.get('answer') and item['answer'].strip())}). Перехожу к генерации финального ответа.")
            return "CONTINUE"
        
        if state['rephrasing_count'] >= self.max_retries:
            print("❌ Достигнут лимит попыток переформулирования запросов и поиска. Завершаю работу.")
            if not state.get('final_answer'): 
                state['final_answer'] = "К сожалению, не удалось найти релевантную информацию или сгенерировать ответы после нескольких попыток."
            return "END"
        
        print(f"⚠️ Поиск и анализ не дали полезных Q&A пар. Попытка №{state['rephrasing_count'] + 1}. Возвращаюсь к переформулированию запросов.")
        return "RETRY"

    def run(self, query: str):
        """
        Запускает выполнение пайплайна.
        """
        initial_state = GraphState(
            original_query=query,
            search_queries=[],
            qa_results=[],
            feedback="",
            rephrasing_count=0,
            final_answer=""
        )
        print(f"Обработка запроса: '{query}'")
        # Если Langfuse активирован, LangfuseHandler внутри LLMHandler сам будет управлять колбэками.
        final_state = self.graph.invoke(initial_state) 

        if not final_state.get('final_answer'):
            final_state['final_answer'] = "Не удалось сгенерировать ответ на основе имеющейся информации."
        self._print_final_result(final_state)
        self._save_results_to_json(final_state)
        return final_state

    def _print_final_result(self, final_state: GraphState):
            """Выводит финальный ответ и сводку выполнения."""
            print("\n" + "#" * 60 + "\n###" + " " * 22 + "ИТОГИ ВЫПОЛНЕНИЯ" + " " * 22 + "###\n" + "#" * 60)
            print("\n" + "="*23 + " ФИНАЛЬНЫЙ ОТВЕТ " + "="*23)
            print(final_state['final_answer'])
            print("="*60)
        
    def _save_results_to_json(self, final_state: GraphState):
        """Сохраняет полное состояние пайплайна в JSON-файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_log_{timestamp}.json"
        state_to_save = final_state.copy()
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            with open(os.path.join(config.DATA_DIR, filename), 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4)
            print(f"\n[ИНФО] Полный лог выполнения сохранен в файл: {filename}")
        except Exception as e:
            print(f"\n[ОШИБКА] Не удалось сохранить лог в файл: {e}")

class ComponentFactory:
    """
    Фабрика для создания и настройки компонентов пайплайна.
    """
    def create_llm_handler(self, model_name: str) -> LLMHandler:
        return LLMHandler(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            model_name=model_name
        )
    
    def create_combined_web_searcher(self) -> CombinedWebSearcher:
        web_searcher_instances: List[BaseSearcher] = [] 
        
        # Проверяем, сконфигурирован ли Yandex Search
        if config.YANDEX_FOLDER_ID != "your_yandex_folder_id_here" and config.YANDEX_OAUTH_TOKEN != "your_yandex_oauth_token_here":
            web_searcher_instances.append(YandexSearcher(
                folder_id=config.YANDEX_FOLDER_ID,
                oauth_token=config.YANDEX_OAUTH_TOKEN
            ))
            print("Yandex Search провайдер активирован.")
        else:
             print("Yandex Search провайдер не сконфигурирован (проверьте YANDEX_FOLDER_ID и YANDEX_OAUTH_TOKEN в config.py).")
             
        # Здесь можно добавить другие поисковые провайдеры (например, Google Searcher)
        # if config.GOOGLE_API_KEY and config.GOOGLE_CSE_ID:
        #     from searchers.google_searcher import GoogleSearcher
        #     web_searcher_instances.append(GoogleSearcher(
        #         api_key=config.GOOGLE_API_KEY,
        #         cse_id=config.GOOGLE_CSE_ID
        #     ))
        #     print("Google Search провайдер активирован.")

        if not web_searcher_instances:
            raise ValueError("Не сконфигурирован ни один внешний поисковый провайдер. Проверьте config.py.")
        
        return CombinedWebSearcher(searchers=web_searcher_instances)

    def create_langgraph_pipeline(self, args: argparse.Namespace) -> LangGraphPipeline:
        """Создает и настраивает экземпляр LangGraphPipeline."""
        llm_handler = self.create_llm_handler(args.model)
        combined_searcher = self.create_combined_web_searcher()
        return LangGraphPipeline(
            llm_handler=llm_handler,
            web_searcher=combined_searcher,
            max_retries=config.MAX_RETRIES 
        )

def main():
    """
    Основная функция для запуска агента LangGraph.
    """
    parser = argparse.ArgumentParser(description="Запрос к LLM с итеративным поиском в вебе через LangGraph.")
    parser.add_argument("--query", default=config.DEFAULT_QUERY, help="Текст запроса")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="Количество результатов для каждого поисковика")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Имя модели LLM")
    args = parser.parse_args()
    
    config.DEFAULT_LIMIT = args.limit # Обновляем лимит по умолчанию из аргументов командной строки

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
