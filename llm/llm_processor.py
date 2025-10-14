# llm/llm_processor.py

import tiktoken
from typing import List
from .llm_handler import LLMHandler # Импортируем ваш существующий LLMHandler



# Промпт для "Map" шага: извлечение релевантной информации из одного чанка
CHUNK_PROCESSOR_PROMPT = """
Из предоставленного ниже текста извлеки и кратко перечисли только ту информацию, которая напрямую относится к запросу пользователя.
Сохраняй ключевые факты, цифры и выводы. Если в тексте нет релевантной информации, верни пустой ответ.

Запрос пользователя: "{query}"

Текст для анализа:
---
{chunk}
---
"""

class LLMProcessor:
    """
    Класс для обработки больших объемов текста перед отправкой в LLM.
    Реализует стратегию Map-Reduce для обхода ограничения контекстного окна.
    """
    def __init__(self, llm_handler: LLMHandler, model_name: str, 
                 max_tokens_per_chunk: int = 4000, model_context_window: int = 16000):
        self.llm_handler = llm_handler
        try:
            # Выбираем кодировщик токенов в зависимости от модели
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print("Warning: Model not found. Using cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.model_context_window = model_context_window

    def _estimate_tokens(self, text: str) -> int:
        """Оценивает количество токенов в строке."""
        return len(self.tokenizer.encode(text))

    def _create_chunks(self, text: str) -> List[str]:
        """Разбивает большой текст на чанки заданного размера в токенах."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens_per_chunk):
            chunk_tokens = tokens[i:i + self.max_tokens_per_chunk]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    def process_large_context(self, final_prompt_template: str, query: str, search_results: str,
                               max_tokens_for_final_answer: int = 4096) -> str:
        """
        Основной метод, реализующий Map-Reduce.
        
        Args:
            final_prompt_template (str): Ваш исходный FINAL_ANSWER_GENERATOR_PROMPT.
            query (str): Запрос пользователя.
            search_results (str): Большой текст с результатами поиска.
            
        Returns:
            str: Финальный ответ от LLM.
        """
        # 1. Оцениваем общий размер search_results
        search_results_tokens = self._estimate_tokens(search_results)
        prompt_template_tokens = self._estimate_tokens(final_prompt_template.format(query=query, search_results=""))
        
        # Если все вместе помещается в контекст, просто вызываем LLM напрямую
        # Оставляем запас ~4096 токенов на ответ
        if (search_results_tokens + prompt_template_tokens) < (self.model_context_window - max_tokens_for_final_answer):
            print("Текст помещается в контекстное окно. Выполняется прямой запрос.")
            final_prompt = final_prompt_template.format(query=query, search_results=search_results)
            return self.llm_handler.get_response(
                prompt=final_prompt, 
                max_tokens=max_tokens_for_final_answer
            )

        # 2. Если не помещается, начинаем процесс Map-Reduce
        print("Текст слишком большой. Запуск процесса Map-Reduce...")
        
        # 3. Разбиваем на чанки (Chunking)
        chunks = self._create_chunks(search_results)
        print(f"Текст разбит на {len(chunks)} чанков.")
        
        # 4. Обрабатываем каждый чанк (Map)
        relevant_info_list = []
        for i, chunk in enumerate(chunks):
            print(f"Обработка чанка {i+1}/{len(chunks)}...")
            map_prompt = CHUNK_PROCESSOR_PROMPT.format(query=query, chunk=chunk)
            
            # Используем get_response для получения выжимки из чанка
            summary = self.llm_handler.get_response(prompt=map_prompt, temperature=0.0, max_tokens=1024)
            if summary:
                relevant_info_list.append(summary)
        
        # 5. Объединяем результаты (Reduce)
        print("Объединение результатов и генерация финального ответа...")
        combined_summaries = "\n\n---\n\n".join(relevant_info_list)
        
        # Проверяем, не превышает ли размер конспекта контекст
        if self._estimate_tokens(combined_summaries) + prompt_template_tokens >= (self.model_context_window - max_tokens_for_final_answer):
            print("Внимание: Даже после обработки чанков итоговый конспект слишком велик. Возвращаем обрезанный конспект.")
            # В реальном приложении здесь можно применить рекурсивную обработку
            # Но для простоты мы просто вернем то, что успели собрать
            final_search_results = combined_summaries
        else:
            final_search_results = combined_summaries

        # 6. Генерируем финальный ответ с использованием исходного промпта
        final_prompt = final_prompt_template.format(query=query, search_results=final_search_results)
         ### ИЗМЕНЕНО: Передаем заданное количество токенов и сюда ###
        return self.llm_handler.get_response(
            prompt=final_prompt, 
            max_tokens=max_tokens_for_final_answer
        )