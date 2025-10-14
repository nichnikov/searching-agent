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
### ДОБАВЛЕНО: Импортируем новый LLMProcessor ###
from llm.llm_processor import LLMProcessor
from llm.llm_handler import LLMHandler
from searchers.base_searcher import BaseSearcher
from searchers.combined_web_searcher import CombinedWebSearcher
from searchers.yandex_searcher import YandexSearcher
from utils.str2dir import structure_text_to_json_list
from langgraph.graph import StateGraph, END
from prompts.templates import (
    SEARCH_QUERY_GENERATOR_PROMPT,
    PER_QUERY_ANALYZER_PROMPT,
    FINAL_ANSWER_GENERATOR_PROMPT,
    CONTENT_COMPRESSION_PROMPT
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
    original_search_query_context: str = ""

# --- Определение состояния графа ---

class GraphState(TypedDict):
    original_query: str
    search_queries: List[str]
    qa_results: List[LLMAnalysis]
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

        ### ДОБАВЛЕНО: Инициализация LLMProcessor ###
        # LLMProcessor будет использовать тот же llm_handler для выполнения запросов.
        # Убедитесь, что в вашем файле config.py есть переменная MODEL_CONTEXT_WINDOW
        # Например: MODEL_CONTEXT_WINDOW = 128000 для 'gpt-4-turbo'
        self.llm_processor = LLMProcessor(
            llm_handler=self.llm_handler,
            model_name=self.llm_handler.model_name,
            model_context_window=config.MODEL_CONTEXT_WINDOW 
        )
        print(f"LLMProcessor инициализирован для модели '{self.llm_handler.model_name}' с окном контекста {config.MODEL_CONTEXT_WINDOW} токенов.")


    def _build_graph(self):
        """Собирает граф LangGraph с узлами и переходами."""
        graph = StateGraph(GraphState)
        graph.add_node("generate_queries", self.generate_search_queries_node)
        graph.add_node("search_and_analyze_per_query", self.search_and_analyze_per_query_node)
        graph.add_node("generate_answer", self.generate_final_answer_node)

        graph.set_entry_point("generate_queries")
        graph.add_edge("generate_queries", "search_and_analyze_per_query")

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
        return {"search_queries": queries, "rephrasing_count": state['rephrasing_count'] + 1}

### ИЗМЕНЕНО: Узел теперь использует компоненты LLMProcessor для сжатия контента ###
    def search_and_analyze_per_query_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Выполняет поиск, сжимает слишком большие документы и анализирует результаты с помощью LLM.
        """
        print("\n" + "="*20 + " ЭТАП 2: ПОИСК И АНАЛИЗ ПО КАЖДОМУ ЗАПРОСУ " + "="*20)

        original_query = state['original_query']
        search_queries = state['search_queries']
        all_qa_results: List[LLMAnalysis] = list(state.get('qa_results', []))
        
        # Устанавливаем порог в токенах для одного документа. Если больше - сжимаем.
        # Это значение можно вынести в config.py
        # CONTENT_TOKEN_THRESHOLD = 4000 

        if not search_queries:
            # ... (код без изменений) ...
            print("Нет поисковых запросов для обработки.")
            return {"feedback": "Не удалось сгенерировать поисковые запросы."}

        for i, single_query in enumerate(search_queries):
            print(f"Обработка поискового запроса [{i+1}/{len(search_queries)}]: '{single_query}'")

            try:
                single_query_search_results = self.web_searcher.search(
                    query=single_query,
                    num_results=config.DEFAULT_LIMIT
                )
            
            except Exception as e:
                # ... (код без изменений) ...
                continue

            if not single_query_search_results:
                # ... (код без изменений) ...
                continue

            ### ДОБАВЛЕНО: Логика предварительной обработки и сжатия контента ###
            processed_search_results = []
            for doc in single_query_search_results:
                content = doc.get("content", "")
                # Используем токенизатор из нашего LLMProcessor для оценки размера
                num_tokens = self.llm_processor._estimate_tokens(content)

                if num_tokens > config.CONTENT_TOKEN_THRESHOLD:
                    print(f"    - Контент из источника '{doc.get('title', 'N/A')}' слишком велик ({num_tokens} токенов). Сжимаем...")
                    
                    compression_prompt = CONTENT_COMPRESSION_PROMPT.format(
                        search_query=single_query,
                        content=content
                    )
                    # Выполняем сжатие с помощью LLM
                    compressed_content = self.llm_handler.get_response(
                        compression_prompt, 
                        temperature=0.0, # Низкая температура для точности
                        max_tokens=1024 # Ограничиваем размер выжимки
                    )
                    
                    # Создаем копию документа с обновленным, сжатым контентом
                    new_doc = doc.copy()
                    new_doc["content"] = compressed_content
                    processed_search_results.append(new_doc)
                    print("    - Сжатие завершено.")
                else:
                    # Если контент в пределах нормы, просто добавляем его
                    processed_search_results.append(doc)

            # Используем обработанный (местами сжатый) список для дальнейшей работы
            formatted_search_answer_blocks = [{
                "title": doc.get("title", ""), "url": doc.get("url", ""), "content": doc.get("content", "")
            } for doc in processed_search_results] # <-- ИЗМЕНЕНО: используется processed_search_results
            formatted_search_answer_string = json.dumps(formatted_search_answer_blocks, ensure_ascii=False, indent=2)

            prompt_for_analyzer = PER_QUERY_ANALYZER_PROMPT.format(query=original_query, search_answer=formatted_search_answer_string)
            response_from_llm = self.llm_handler.get_response(prompt_for_analyzer, response_format={"type": "json_object"})

            try:
                # ... (остальная часть функции без изменений) ...
                extracted_json_list = structure_text_to_json_list(response_from_llm)
                
                if not extracted_json_list:
                    print(f"⚠️ Функция structure_text_to_json_list не смогла извлечь JSON для запроса '{single_query}'. Пропускаю.")
                    continue

                for item_dict in extracted_json_list:
                    item_dict["original_search_query_context"] = single_query
                    all_qa_results.append(item_dict)
                    
                print(f"✅ Анализ для запроса '{single_query}' выполнен. Обработано {len(extracted_json_list)} объектов.")
                print(f"Собрано {len(extracted_json_list)} новых Q&A пар. Всего: {len(all_qa_results)}.")

            except Exception as e:
                print(f"❌ Непредвиденная ошибка при обработке ответа LLM для запроса '{single_query}': {e}. Пропускаю.")
            
        return {"qa_results": all_qa_results}
    

    ### ИЗМЕНЕНО: Логика генерации финального ответа теперь использует LLMProcessor ###
    def generate_final_answer_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Генерирует финальный ответ, используя LLMProcessor для обработки потенциально
        большого объема собранной информации.
        """
        print("\n" + "="*20 + " ЭТАП 3: ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА " + "="*20)
        original_query = state['original_query']
        qa_results = state.get('qa_results', [])

        if not qa_results:
            return {"final_answer": "Не удалось сгенерировать ответ на основе найденной информации."}

        # Шаг 1: Формируем большой строковый контекст из всех результатов, как и раньше.
        formatted_qa_for_final_answer = ""
        source_counter = 1
        for qa_item in qa_results:
            for data_source in qa_item.get("data", []):
                formatted_qa_for_final_answer += f"--- Источник {source_counter} (поисковый запрос: '{str(qa_item.get('original_search_query_context', ''))}') ---\n"
                formatted_qa_for_final_answer += f"Заголовок: {str(data_source.get('title', ''))}\n"
                formatted_qa_for_final_answer += f"Ссылка: {str(data_source.get('url', ''))}\n"
                formatted_qa_for_final_answer += f"Краткий ответ по этому источнику: {str(qa_item.get('answer', ''))}\n"
                formatted_qa_for_final_answer += f"Фргамент текста, на базе которого сформулирован краткий ответ: {str(data_source.get('fragment', ''))}\n\n"
                source_counter += 1

        if not formatted_qa_for_final_answer.strip():
            print("Все Q&A пары оказались пустыми или нерелевантными.")
            return {"final_answer": "Не удалось сгенерировать содержательный ответ."}

        # Шаг 2: Вместо прямого вызова LLM, передаем задачу LLMProcessor.
        # Он сам определит, нужно ли разбивать текст на чанки, и вернет готовый ответ.
        print("Передача собранной информации в LLMProcessor для генерации финального ответа...")
        answer = self.llm_processor.process_large_context(
            final_prompt_template=FINAL_ANSWER_GENERATOR_PROMPT,
            query=original_query,
            search_results=formatted_qa_for_final_answer,
            # Передаем сюда увеличенное значение из конфига
            max_tokens_for_final_answer=config.MAX_TOKENS_FINAL_ANSWER 
        )
        
        print(f"✅ Финальный ответ сгенерирован (запрошенный лимит токенов: {config.MAX_TOKENS_FINAL_ANSWER}).")
        return {"final_answer": answer}

    def decide_next_step(self, state: GraphState) -> str:
        """Определяет следующий шаг: продолжить, повторить или завершить."""
        print("\n" + "="*20 + " ПРИНЯТИЕ РЕШЕНИЯ " + "="*20)
        qa_results = state.get('qa_results', [])
        normalized_qa_results = []
        
        for qa in qa_results:
            try:
                if isinstance(qa, LLMAnalysis):
                    normalized_qa_results.append(qa)
                else:
                    validated_qa = LLMAnalysis(**qa)
                    normalized_qa_results.append(validated_qa)
            except ValidationError as e:
                print(f"⚠️ Ошибка валидации одного из Q&A результатов. Элемент будет проигнорирован. Ошибка: {e}")
                print(f"   Проблемный элемент: {qa}")
                continue

        state['qa_results'] = normalized_qa_results
        
        meaningful_qa_found = any(
            qa.answer.strip() and any(ds.url.strip() for ds in qa.data) for qa in normalized_qa_results
        )

        if meaningful_qa_found:
            print(f"✅ Найдено {len(normalized_qa_results)} содержательных Q&A пар. Перехожу к генерации финального ответа.")
            return "CONTINUE"

        if state['rephrasing_count'] >= self.max_retries:
            print("❌ Достигнут лимит попыток. Завершаю работу.")
            return "END"
        
        feedback = "Предыдущие поисковые запросы не дали релевантных результатов. Попробуй сгенерировать запросы под другим углом, используя другие ключевые слова."
        print(f"⚠️ Не найдено полезных Q&A пар. Попытка №{state['rephrasing_count'] + 1}. Возвращаюсь к переформулированию запросов.")
        state['feedback'] = feedback
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
        
        state_to_save = final_state.copy()
        state_to_save['qa_results'] = [item.dict() if isinstance(item, BaseModel) else item for item in state_to_save.get('qa_results', [])]

        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            with open(os.path.join(config.DATA_DIR, filename), 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4, default=str)
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
        return LangGraphPipeline(llm_handler=llm_handler, web_searcher=combined_searcher, max_retries=args.retries)

def main():
    """Основная функция для запуска агента LangGraph."""
    parser = argparse.ArgumentParser(description="Запрос к LLM с итеративным поиском в вебе через LangGraph.")
    parser.add_argument("--query", default=config.DEFAULT_QUERY, help="Текст запроса")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="Количество результатов для каждого поисковика")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Имя модели LLM")
    parser.add_argument("--retries", type=int, default=config.MAX_RETRIES, help="Максимальное количество попыток переформулирования запросов")
    args = parser.parse_args()
    
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