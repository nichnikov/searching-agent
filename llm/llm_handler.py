from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

class LLMHandler:
    """Класс для работы с языковой моделью."""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        if not all([base_url, api_key, model_name]):
            raise ValueError("Необходимо указать base_url, api_key и model_name для LLM.")
        try:
            self.llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model_name)
            print(f"LLM-клиент для модели '{model_name}' успешно инициализирован.")
        except Exception as e:
            print(f"Ошибка при инициализации LLM-клиента: {e}")
            raise

    def get_response(self, prompt: str) -> str:
        """
        Отправляет промпт модели и возвращает ее ответ.

        Args:
            prompt: Текстовый промпт для модели.

        Returns:
            Ответ модели в виде строки.
        """
        try:
            messages = [HumanMessage(content=prompt)]
            ai_msg = self.llm.invoke(messages)
            return ai_msg.content
        except Exception as e:
            print(f"Произошла ошибка при обращении к LLM: {e}")
            return "Ошибка при получении ответа от языковой модели."