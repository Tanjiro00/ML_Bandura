from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import time
import random
import re


def search_text(query: str, num_results: int = 5) -> list[dict]:
    """
    Ищет веб-страницы по запросу и возвращает чистый текст контента.

    Параметры:
        query (str): Поисковый запрос (может быть вопросом)
        num_results (int): Количество возвращаемых результатов

    Возвращает:
        list[dict]: Список словарей с URL и текстовым контентом
    """
    with DDGS() as ddgs:
        # Получаем и фильтруем результаты поиска
        results = []
        for result in ddgs.text(query, region='ru-ru', max_results=num_results*2):
            url = result.get('href', '')
            if not re.search(r'\.(pdf|doc|docx|xls|ppt|pptx)$', url, re.I):
                results.append(url)
            if len(results) >= num_results*2:
                break

    # Собираем текстовый контент
    output = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'ru-RU,ru;q=0.9'
    }

    for url in results[:num_results]:
        try:
            time.sleep(random.uniform(0.5, 2.5))
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            if 'text/html' not in response.headers.get('Content-Type', ''):
                continue

            # Извлекаем и очищаем текст
            soup = BeautifulSoup(response.text, 'html.parser')

            # Удаляем ненужные элементы
            for tag in soup(['script', 'style', 'meta', 'link', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            # Получаем чистый текст с форматированием
            text = soup.get_text(separator='\n', strip=True)

            # Дополнительная очистка
            text = re.sub(r'\n{3,}', '\n\n', text)  # Удаляем множественные переносы
            text = re.sub(r'[\xa0]+', ' ', text)     # Удаляем неразрывные пробелы
            text = text[:15000]                      # Ограничиваем размер текста

            output.append({
                'url': url,
                'text': text,
                'status': 'success',
                'content_length': len(text)
            })

        except Exception as e:
            output.append({
                'url': url,
                'status': 'error',
                'error': str(e)
            })

    return output