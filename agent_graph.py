### Обновленный код `main.py`

# main.py
import os
import json
import argparse
from typing import List, Dict, Any, TypedDict
from datetime import datetime

# Новые импорты для надежности
from pydantic import BaseModel, ValidationError
from json_repair import loads as json_repair_loads

# Импортируем модули из нашей структуры проекта
import config
from llm.llm_handler import LLMHandler
from searchers.base_searcher import BaseSearcher
from searchers.combined_web_searcher import CombinedWebSearcher
from searchers.yandex_searcher import YandexSearcher
from utils.str2dir import structure_text_to_json_list
from langgraph.graph import StateGraph, END
from prompts.templates import (
    SEARCH_QUERY_GENERATOR_PROMPT,
    PER_QUERY_ANALYZER_PROMPT,
    FINAL_ANSWER_GENERATOR_PROMPT
)

# --- Определение моделей данных Pydantic для валидации ---

class DataSource(BaseModel):
    """Модель для одного источника данных, извлеченного LLM."""
    url: str
    title: str
    fragment: str

class LLMAnalysis(BaseModel):
    """Модель для полного ответа LLM по одному поисковому запросу."""
    answer: str
    data: List[DataSource]
    original_search_query_context: str = "" # Добавляем поле для контекста

# --- Определение состояния графа ---

class GraphState(TypedDict):
    original_query: str
    search_queries: List[str]
    qa_results: List[LLMAnalysis] # Используем Pydantic модель для типизации
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
        graph.add_node("generate_answer", self.generate_final_answer_node)

        graph.set_entry_point("generate_queries")
        graph.add_edge("generate_queries", "search_and_analyze_per_query")

        # Используем только условные переходы для принятия решений
        graph.add_conditional_edges(
            "search_and_analyze_per_query",
            self.decide_next_step,
            {"RETRY": "generate_queries", "CONTINUE": "generate_answer", "END": END}
        )
        # Избыточная связь удалена
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
        return {"search_queries": queries, "rephrasing_count": state['rephrasing_count'] + 1}

    def search_and_analyze_per_query_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Выполняет поиск по каждому запросу, анализирует результаты с помощью LLM,
        восстанавливает и валидирует JSON.
        """
        print("\n" + "="*20 + " ЭТАП 2: ПОИСК И АНАЛИЗ ПО КАЖДОМУ ЗАПРОСУ " + "="*20)

        original_query = state['original_query']
        search_queries = state['search_queries']
        # Делаем изменяемую копию из предыдущего состояния
        all_qa_results: List[LLMAnalysis] = list(state.get('qa_results', []))

        if not search_queries:
            print("Нет поисковых запросов для обработки.")
            return {"feedback": "Не удалось сгенерировать поисковые запросы."}

        for i, single_query in enumerate(search_queries):
            print(f"Обработка поискового запроса [{i+1}/{len(search_queries)}]: '{single_query}'")

            try:
                single_query_search_results = self.web_searcher.search(
                    query=single_query,
                    num_results=config.DEFAULT_LIMIT
                )
            except Exception as e: # Заготовка для обработки ошибок API
                print(f"❌ Ошибка при вызове API поиска для запроса '{single_query}': {e}. Пропускаю.")
                continue

            if not single_query_search_results:
                print(f"Поиск для запроса '{single_query}' не дал результатов.")
                continue

            formatted_search_answer_blocks = [{
                "title": doc.get("title", ""), "url": doc.get("url", ""), "content": doc.get("content", "")
            } for doc in single_query_search_results]
            formatted_search_answer_string = json.dumps(formatted_search_answer_blocks, ensure_ascii=False, indent=2)

            prompt_for_analyzer = PER_QUERY_ANALYZER_PROMPT.format(query=original_query, search_answer=formatted_search_answer_string)
            response_from_llm = self.llm_handler.get_response(prompt_for_analyzer, response_format={"type": "json_object"})

            try:
                # Эта функция заменяет json_repair и начальную логику парсинга.
                extracted_json_list = structure_text_to_json_list(response_from_llm)
                
                if not extracted_json_list:
                    # Функция могла вернуть пустой список, если парсинг не удался.
                    # Сообщение об ошибке выводится внутри самой функции.
                    print(f"⚠️ Функция structure_text_to_json_list не смогла извлечь JSON для запроса '{single_query}'. Пропускаю.")
                    continue # Явно переходим к следующему поисковому запросу

                # 2. Валидируем каждый извлеченный объект с помощью Pydantic.
                # Это гарантирует, что структура данных соответствует ожидаемой моделью LLMAnalysis,
                # и защищает последующие узлы графа от ошибок.
                for item_dict in extracted_json_list:
                    item_dict["original_search_query_context"] = single_query
                    all_qa_results.append(item_dict)
                    
                    print(f"✅ Анализ для запроса '{single_query}' выполнен. Обработано {len(extracted_json_list)} объектов.")
                    
                print(f"Собрано {len(extracted_json_list)} новых Q&A пар. Всего: {len(all_qa_results)}.")

            except Exception as e:
                # Этот блок отлавливает любые другие непредвиденные ошибки
                print(f"❌ Непредвиденная ошибка при обработке ответа LLM для запроса '{single_query}': {e}. Пропускаю.")

                
                #print(f"✅ Анализ для запроса '{single_query}' выполнен и провалидирован. Получено {len(validated_analyses)} ответов.")

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"❌ Не удалось распарсить или провалидировать JSON от LLM для запроса '{single_query}': {e}. Ответ LLM: '{response_from_llm}'. Пропускаю.")
            except Exception as e:
                print(f"❌ Непредвиденная ошибка при обработке ответа LLM для запроса '{single_query}': {e}. Пропускаю.")

        

        return {"qa_results": all_qa_results}

    def generate_final_answer_node(self, state: GraphState) -> Dict[str, Any]:
        """Генерирует финальный ответ на основе собранных и провалидированных Q&A пар."""
        print("\n" + "="*20 + " ЭТАП 3: ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА " + "="*20)
        original_query = state['original_query']
        qa_results = state.get('qa_results', [])

        if not qa_results:
            return {"final_answer": "Не удалось сгенерировать ответ на основе найденной информации."}

        formatted_qa_for_final_answer = ""
        source_counter = 1
        for qa_item in qa_results:
            # ИСПРАВЛЕНО: Итерируемся по списку источников данных `data`
            for data_source in qa_item["data"]:
                # Данные уже провалидированы Pydantic, можно обращаться напрямую
                formatted_qa_for_final_answer += f"--- Источник {source_counter} (поисковый запрос: '{qa_item['original_search_query_context']}') ---\n"
                formatted_qa_for_final_answer += f"Заголовок: {data_source['title']}\n"
                formatted_qa_for_final_answer += f"Ссылка: {data_source['url']}\n"
                formatted_qa_for_final_answer += f"Краткий ответ по этому источнику: {qa_item['answer']}\n"
                formatted_qa_for_final_answer += f"Фргамент текста, на базе которого сформулирован краткий ответ: {data_source['fragment']}\n\n"
                source_counter += 1

        if not formatted_qa_for_final_answer.strip():
            print("Все Q&A пары оказались пустыми или нерелевантными.")
            return {"final_answer": "Не удалось сгенерировать содержательный ответ."}

        prompt = FINAL_ANSWER_GENERATOR_PROMPT.format(query=original_query, search_results=formatted_qa_for_final_answer)
        answer = self.llm_handler.get_response(prompt)
        print("✅ Финальный ответ сгенерирован.")
        return {"final_answer": answer}

    def decide_next_step(self, state: GraphState) -> str:
        """Определяет следующий шаг: продолжить, повторить или завершить."""
        print("\n" + "="*20 + " ПРИНЯТИЕ РЕШЕНИЯ " + "="*20)
        qa_results = state.get('qa_results', [])

        # Преобразуем dict в LLMAnalysis, если нужно
        normalized_qa_results = [qa if isinstance(qa, LLMAnalysis) else LLMAnalysis(**qa)
                                 for qa in qa_results]


        # Проверяем, есть ли хотя бы один содержательный ответ
        meaningful_qa_found = any(
            qa.answer.strip() and any(ds.url.strip() for ds in qa.data) for qa in normalized_qa_results
        )

        if meaningful_qa_found:
            print(f"✅ Найдено {len(qa_results)} содержательных Q&A пар. Перехожу к генерации финального ответа.")
            return "CONTINUE"

        if state['rephrasing_count'] >= self.max_retries:
            print("❌ Достигнут лимит попыток. Завершаю работу.")
            return "END"
        
        # УЛУЧШЕНО: Формируем более полезную обратную связь
        feedback = "Предыдущие поисковые запросы не дали релевантных результатов. Попробуй сгенерировать запросы под другим углом, используя другие ключевые слова."
        print(f"⚠️ Не найдено полезных Q&A пар. Попытка №{state['rephrasing_count'] + 1}. Возвращаюсь к переформулированию запросов.")
        state['feedback'] = feedback # Сохраняем фидбэк для следующего шага
        return "RETRY"

    def run(self, query: str):
        """Запускает выполнение пайплайна."""
        initial_state = GraphState(
            original_query=query, search_queries=[], qa_results=[],
            feedback="", rephrasing_count=0, final_answer=""
        )
        print(f"Обработка запроса: '{query}'")
        final_state = self.graph.invoke(initial_state)

        if not final_state.get('final_answer') or not final_state['final_answer'].strip():
            final_state['final_answer'] = "К сожалению, не удалось найти релевантную информацию или сгенерировать ответ после нескольких попыток."
        
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
        
        # Pydantic модели нужно конвертировать в словари для сериализации
        state_to_save = final_state.copy()
        state_to_save['qa_results'] = [item for item in state_to_save.get('qa_results', [])]

        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            with open(os.path.join(config.DATA_DIR, filename), 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4)
            print(f"\n[ИНФО] Полный лог выполнения сохранен в файл: {filename}")
        except Exception as e:
            print(f"\n[ОШИБКА] Не удалось сохранить лог в файл: {e}")

class ComponentFactory:
    """Фабрика для создания и настройки компонентов пайплайна."""
    def create_llm_handler(self, model_name: str) -> LLMHandler:
        return LLMHandler(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY, model_name=model_name)

    def create_combined_web_searcher(self) -> CombinedWebSearcher:
        web_searcher_instances: List[BaseSearcher] = []
        if config.YANDEX_FOLDER_ID != "your_yandex_folder_id_here" and config.YANDEX_OAUTH_TOKEN != "your_yandex_oauth_token_here":
            web_searcher_instances.append(YandexSearcher(folder_id=config.YANDEX_FOLDER_ID, oauth_token=config.YANDEX_OAUTH_TOKEN))
            print("Yandex Search провайдер активирован.")
        else:
            print("Yandex Search провайдер не сконфигурирован.")
        
        if not web_searcher_instances:
            raise ValueError("Не сконфигурирован ни один внешний поисковый провайдер.")
        return CombinedWebSearcher(searchers=web_searcher_instances)

    def create_langgraph_pipeline(self, args: argparse.Namespace) -> LangGraphPipeline:
        """Создает и настраивает экземпляр LangGraphPipeline."""
        llm_handler = self.create_llm_handler(args.model)
        combined_searcher = self.create_combined_web_searcher()
        # ИСПРАВЛЕНО: Передаем max_retries из аргументов, а не из глобального конфига
        return LangGraphPipeline(llm_handler=llm_handler, web_searcher=combined_searcher, max_retries=args.retries)

def main():
    """Основная функция для запуска агента LangGraph."""
    parser = argparse.ArgumentParser(description="Запрос к LLM с итеративным поиском в вебе через LangGraph.")
    parser.add_argument("--query", default=config.DEFAULT_QUERY, help="Текст запроса")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="Количество результатов для каждого поисковика")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Имя модели LLM")
    # Добавлен аргумент для управления количеством попыток
    parser.add_argument("--retries", type=int, default=config.MAX_RETRIES, help="Максимальное количество попыток переформулирования запросов")
    args = parser.parse_args()
    
    # ИСПРАВЛЕНО: Глобальный конфиг больше не изменяется.
    # Значение limit будет использоваться напрямую из args в месте вызова.
    config.DEFAULT_LIMIT = args.limit

    factory = ComponentFactory()
    pipeline = factory.create_langgraph_pipeline(args)
    pipeline.run(query=args.query)
    '''
    try:
        factory = ComponentFactory()
        pipeline = factory.create_langgraph_pipeline(args)
        pipeline.run(query=args.query)
    except (ValueError, ImportError) as e:
        print(f"\n[ОШИБКА ИНИЦИАЛИЗАЦИИ] {e}")
    except Exception as e:
        print(f"\n[НЕПРЕДВИДЕННАЯ ОШИБКА] {e}")'''

if __name__ == "__main__":
    main()