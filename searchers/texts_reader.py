import newspaper

class SiteTextExtractor:
    """
    Класс для извлечения основного текста со списка веб-страниц.
    """

    def __init__(self):
        """
        Инициализация класса.
        """
        pass

    def fetch_texts(self, urls):
        """
        Получает основной текст со списка URL-адресов.

        Args:
            urls (list): Список URL-адресов в виде строк.

        Returns:
            dict: Словарь, где ключами являются URL-адреса,
                  а значениями - извлеченный текст или сообщение об ошибке.
        """
        results = []
        for url in urls:
            try:
                # Создаем объект статьи
                article = newspaper.Article(url)

                # Загружаем HTML-содержимое страницы
                article.download()

                # Анализируем (парсим) страницу для извлечения текста
                article.parse()

                # Сохраняем извлеченный текст
                results.append({"url": url, "text": article.text})
                # results[url] = article.text
            except Exception as e:
                # В случае ошибки сохраняем сообщение
                results.append({"url": url, "text": f"Не удалось получить текст: {e}"})
                # results[url] = f"Не удалось получить текст: {e}"
        return results

# Пример использования класса

if __name__ == '__main__':
    # Список URL-адресов для обработки
    url_list = [
        'https://kontur.ru/market/spravka/53707-biznesu_na_usn_vydavat_cheki_s_nds',
        'https://www.glavbukh.ru/news/48659-fns-rasskazala-kak-uproshchentsam-vybivat-cheki-bez-nds-v-2025-godu',
        'https://www.nalog.gov.ru/rn03/news/activities_fts/15418336/',
        'https://gendalf.ru/news/kassy/stavki-sdelany-kak-pravilno-rabotat-s-nd/',
        'https://www.klerk.ru/blogs/platformaofd/637385/'
    ]

    # Создаем экземпляр класса
    extractor = SiteTextExtractor()

    # Получаем тексты сайтов
    extracted_texts = extractor.fetch_texts(url_list)

    # Выводим результаты
    for d in extracted_texts:
        print(f"URL:", d["url"])
        print("-" * 20)
        # print(f"{text[:500]}...")  # Выводим первые 500 символов для краткости
        txt = d["text"][:500]
        print(f"{txt}...")  # Выводим первые 500 символов для краткости
        print("\n" + "=" * 50 + "\n")