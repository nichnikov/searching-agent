import re
import json
import string
from typing import List, Dict, Union


def clean_string_except_letters_digits_spaces_punctuation(text: str) -> str:
    escaped_punctuation = re.escape(string.punctuation)
    pattern = fr'[^\w \t{escaped_punctuation}]'
    cleaned_text = re.sub(pattern, '', text, flags=re.UNICODE)
    return cleaned_text

# --- НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ОБРАБОТКИ СЛОВАРЯ ---
def _process_dict_lists_to_strings(input_dict: Dict) -> Dict:
    """
    Вспомогательная функция для преобразования значений-списков в словаре в
    строки, где элементы списка разделены ', '.
    Предполагается, что списки содержат только строковые элементы, если их нужно объединить.
    """
    processed_dict = {}
    for key, value in input_dict.items():
        # Если значение является списком и все его элементы - строки, объединяем их
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            processed_dict[key] = ", ".join(value)
        else:
            processed_dict[key] = value
    return processed_dict


# --- ОБНОВЛЕННАЯ ФУНКЦИЯ ДЛЯ СТРУКТУРИРОВАНИЯ JSON ---
def structure_text_to_json_list(text_input: str) -> List[Dict]:
    """
    Анализирует входной текст (предположительно, сырой ответ LLM) и извлекает из него JSON-объекты.

    Сначала очищает текст, затем пытается найти один JSON-объект или список JSON-объектов.
    Если найден один объект, оборачивает его в список.
    Если найдена последовательность объектов "{...}, {...}" без внешних скобок,
    попытается обернуть ее в скобки для корректного парсинга.

    Args:
        text_input (str): Входной текст, предположительно содержащий JSON.

    Returns:
        List[Dict]: Список JSON-объектов (словарей).
                    Возвращает пустой список, если JSON не найден или не может быть корректно разобран.
    """
    # 1. Сначала очищаем весь ответ, так как посторонний текст может быть в начале или конце
    cleaned_input = clean_string_except_letters_digits_spaces_punctuation(text_input)
    cleaned_input_stripped = cleaned_input.strip() # Дополнительно убираем пробелы по краям

    json_to_parse = ""

    # 2. Попытка найти один JSON-объект (ПРИОРИТЕТ)
    # Это важно для вашей структуры, чтобы сначала найти весь объект {...},
    # а не внутренний список [...].
    start_obj = cleaned_input_stripped.find('{')
    end_obj = cleaned_input_stripped.rfind('}')

    # 3. Попытка найти список JSON-объектов (вторичный приоритет)
    start_list = cleaned_input_stripped.find('[')
    end_list = cleaned_input_stripped.rfind(']')

    # Измененная логика: сначала проверяем на одиночный объект, затем на список
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        # Если найден одиночный объект, извлекаем его целиком
        json_to_parse = cleaned_input_stripped[start_obj : end_obj + 1]
    elif start_list != -1 and end_list != -1 and end_list > start_list:
        # Если одиночный объект не найден, но найден список, извлекаем список
        json_to_parse = cleaned_input_stripped[start_list : end_list + 1]
    else:
        # Если ни список, ни одиночный объект не найдены
        print(f"Ошибка: Не найден JSON-объект или список JSON-объектов в очищенном ответе. "
              f"Очищенный текст: {cleaned_input_stripped[:200]}...")
        return []

    # 4. Попытка парсинга извлеченной JSON-строки
    try:
        data: Union[Dict, List[Dict]] = json.loads(json_to_parse)

        processed_results: List[Dict] = []

        if isinstance(data, dict):
            # Если это одиночный словарь, обрабатываем его и добавляем в результат
            processed_results.append(_process_dict_lists_to_strings(data))
        elif isinstance(data, list):
            # Если это список, обрабатываем каждый словарь в нем
            for item in data:
                if isinstance(item, dict):
                    processed_results.append(_process_dict_lists_to_strings(item))
                else:
                    print(f"Ошибка: Список JSON содержит не-словарные элементы. Возвращен пустой список. "
                          f"Извлеченный JSON: {json_to_parse[:200]}...")
                    return []
        else:
            print(f"Ошибка: Неожиданный тип корневого элемента JSON: {type(data)}. Ожидается dict или list. "
                  f"Извлеченный JSON: {json_to_parse[:200]}...")
            return []
        
        return processed_results

    except json.JSONDecodeError as e:
        # 5. Если парсинг не удался, но строка похожа на "{...}, {...}" (последовательность объектов без внешних скобок)
        if json_to_parse.startswith('{') and json_to_parse.endswith('}'):
            modified_text_input = f"[{json_to_parse}]"
            try:
                data_modified: Union[Dict, List[Dict]] = json.loads(modified_text_input)
                
                processed_results_modified: List[Dict] = []
                if isinstance(data_modified, list):
                    for item in data_modified:
                        if isinstance(item, dict):
                            # Применяем обработку к каждому словарю в исправленном списке
                            processed_results_modified.append(_process_dict_lists_to_strings(item))
                        else:
                            print(f"Ошибка: После оборачивания в скобки получен некорректный тип данных (не список словарей). "
                                  f"Извлеченный JSON: {json_to_parse[:200]}...")
                            return []
                    return processed_results_modified
                else:
                    print(f"Ошибка: После оборачивания в скобки получен некорректный тип данных. Ожидался список словарей. "
                          f"Получен: {type(data_modified)}. Извлеченный JSON: {json_to_parse[:200]}...")
                    return []
            except json.JSONDecodeError as e_modified:
                print(f"Ошибка декодирования JSON после оборачивания в скобки: {e_modified}. "
                      f"Извлеченный JSON: {json_to_parse[:200]}...")
                return []
        
        # Если не удалось исправить или это не тот случай, сообщаем об исходной ошибке
        print(f"Ошибка декодирования JSON: {e}. Извлеченный JSON: {json_to_parse[:200]}...")
        return []
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при структурировании JSON: {e}. "
              f"Извлеченный JSON: {json_to_parse[:200]}...")
        return []
    

if __name__ == "__main__":
    txt = """
'{
  "answer": "Компания вправе досрочно перейти на ФСБУ 9/2025 с 1 января 2025 года, закрепив это решение в учетной политике. Стандарт применяется с начала отчетного года, а не с даты его утверждения в середине года.\n\nПереходные проводки отражаются на дату начала применения стандарта, то есть на 1 января 2025 года (на практике это может быть сделано записью на 31 декабря 2024 года). Корректировки проводятся через счет 84 «Нераспределенная прибыль (непокрытый убыток)» в корреспонденции со счетами активов и обязательств, которые затрагивает переход. Счет 90 «Продажи» для корректировок при переходе не используется.\n\nЕсли компания приняла решение о досрочном переходе с 1 января 2025 года и сдает промежуточную отчетность, то она обязана формировать всю отчетность 2025 года, включая отчетность за 9 месяцев, по правилам нового ФСБУ 9/2025. Применять ПБУ 9/99 для промежуточной отчетности, а затем переделывать учет за год, неправомерно.",
  "data": [
    {
      "url": "https://www.buhgalteria.ru/article/fsbu-9-2025-dokhody-perechen-novshestv-dlya-prinyatiya-resheniya-o-dosrochnom-primenenii-standarta",
      "title": "ФСБУ 9/2025 «Доходы»: перечень новшеств для принятия решения о досрочном применении стандарта - Бухгалтерия.ru",
      "fragment": "Организация вправе принять решение о досрочном применении стандарта. Переход на новый порядок в отчетности нужно отражать ретроспективно - так, как если бы новые правила применялись всегда (это общий случай). При этом разрешен и упрощенный порядок."
    },
    {
      "url": "https://www.glavbukh.ru/art/391660-minfin-utverdil-fsbu-92025-dohody-kak-pereyti-i-primenyat",
      "title": "Утвержденный ФСБУ 9/2025 доходы: изменения, отличия, как применять",
      "fragment": "Метод 2. Ретроспективно, но в упрощенном порядке (по выбору организации). ... Последствия перехода отражаются единовременной корректировкой нераспределенной прибыли и соответствующих статей баланса на начало года, с которого применяется стандарт (обычно в межотчетную дату 31 декабря)."
    },
    {
      "url": "https://www.klerk.ru/blogs/klerk365/662405/",
      "title": "ФСБУ 9/2025: как перейти на новый порядок учета доходов",
      "fragment": "На дату начала применения нового ФСБУ нужно скорректировать остатки по следующим счетам бухгалтерского учета: 62 «Расчеты с покупателями и заказчиками»; 76 «Расчеты с разными дебиторами и кредиторами»; 84 «Нераспределенная прибыль (непокрытый убыток)»."
    },
    {
      "url": "https://buh.ru/articles/fsbu-9-2025-dokhody-novovvedeniya-kogda-primenyat-i-kak-podgotovitsya-k-perekhodu.html",
      "title": "ФСБУ 9/2025 «Доходы»: нововведения, когда применять и как подготовиться к переходу | БУХ.1С - сайт для современного бухгалтера",
      "fragment": "Широкий круг организаций в силу закона обязан готовить промежуточную отчетность ежеквартально. Для них отчетным периодом будут I квартал, полугодие, 9 месяцев и отчетный год. ... Иными словами, к каждой отчетной дате необходимо учесть договоры, согласно которым передача контроля производится не единомоментно."
    }
  ]
}'
"""

    res = structure_text_to_json_list(txt)
    print(res)