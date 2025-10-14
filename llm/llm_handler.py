# llm/llm_handler.py
import os
import json
from typing import Dict, Any, Optional

# from langfuse import Langfuse # Раскомментируйте, если используете Langfuse
# from langfuse.callback import CallbackHandler # Раскомментируйте, если используете Langfuse
from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage

class LLMHandler:
    """
    Класс для обработки запросов к LLM (в данном случае, OpenAI).
    """
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        # self.langfuse_handler: Optional[CallbackHandler] = self._init_langfuse() # Раскомментируйте, если используете Langfuse

    # def _init_langfuse(self) -> Optional[CallbackHandler]: # Раскомментируйте, если используете Langfuse
    #     """Инициализирует Langfuse CallbackHandler."""
    #     import config # Импортируем config здесь, чтобы избежать циклического импорта
    #     if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
    #         langfuse = Langfuse(
    #             public_key=config.LANGFUSE_PUBLIC_KEY,
    #             secret_key=config.LANGFUSE_SECRET_KEY,
    #             host=config.LANGFUSE_HOST
    #         )
    #         print("Langfuse инициализирован.")
    #         return CallbackHandler(langfuse)
    #     else:
    #         print("Langfuse не сконфигурирован. Пропускаем инициализацию.")
    #         return None

    def get_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 25000, 
                     response_format: Optional[Dict[str, str]] = None) -> str:
        """
        Отправляет запрос к LLM и возвращает ответ.
        
        Args:
            prompt (str): Текст запроса к LLM.
            temperature (float): Температура генерации (креативность).
            max_tokens (int): Максимальное количество токенов в ответе.
            response_format (Optional[Dict[str, str]]): Формат ответа, например, {"type": "json_object"}.
        
        Returns:
            str: Сгенерированный ответ LLM. В случае ошибки возвращает пустую строку или JSON с ошибкой.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                params["response_format"] = response_format
            
            # if self.langfuse_handler: # Раскомментируйте, если используете Langfuse
            #     params["callbacks"] = [self.langfuse_handler]

            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Ошибка получения ответа от LLM: {e}")
            # Возвращаем структурированную ошибку, если ожидался JSON, иначе пустую строку
            if response_format and response_format.get("type") == "json_object":
                return json.dumps({"error": str(e), "answer": "", "data": {"urls": "", "title": "", "fragment": ""}})
            return ""