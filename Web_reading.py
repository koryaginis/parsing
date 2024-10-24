import sys
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.request import urlopen
import nltk
import glob
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from mpi4py import MPI
from operator import attrgetter

# Класс для хранения результатов вычислений
class CityItem:
    name: str # Название города
    freq: int # Частота встречаемости названия города на странице
    proper_nouns_count: int # Количество собственных имен на странице

    # Инициализация атрибутов класса при создании его объекта
    def __init__(self, name: str, freq: int, proper_nouns_count: int):
        self.name = name
        self.freq = freq
        self.proper_nouns_count = proper_nouns_count

    # Переопределение метода str для представления объекта в виде строки
    def __str__(self):
        return self.name + '\nЧастота встречаемости = ' + str(self.freq) + '\nКоличество собственных имен = ' + str(self.proper_nouns_count)

# Функция возвращениия объекта, который содержит больше всего наименований своего города
def max_by_freq(cities):
    return max(cities, key=attrgetter('freq'))

# Функция возвращениия объекта с наибольшим количеством собственных имен
def max_by_proper_nouns_count(cities):
    return max(cities, key=attrgetter('proper_nouns_count'))

# Функция возвращениия объекта с наименьшим количеством собственных имен
def min_by_proper_nouns_count(cities):
    return min(cities, key=attrgetter('proper_nouns_count'))

# Функция получения содержания html-страницы и очистка от ненужных блоков
def get_text_from_html(fileName):

    # Открытие html-страницы для чтения
    f = open(fileName, "r", encoding="utf8", errors="ignore")

    # Передача содержимого html-страницы в переменную
    html = f.read()

    # Создание объекта класса BeautifulSoup для работы с html-страницами
    soup = BeautifulSoup(html, 'html.parser')

    # Получение основного содержимого html-страницы
    content = soup.find(id='content')

    # Удаление информационных блоков
    content.find(class_='infobox').decompose()

    # Удаление блока примечаний к статье
    for element in content.find(id='Примечания').parent.find_all_next():
        element.decompose()

    # Удаление ссылок для редактирования разделов страницы
    for element in content.find_all(class_='mw-editsection'):
        element.decompose()

    # Возвращение содержания страницы после удаления ненужных блоков
    return content

# Функция возвращения текстового содержимого страницы без бокового меню
def get_main_text(content):
    return content.find(id='mw-content-text').text

# Функция получения названия города, о котором идет речь в полученной статье
def get_city(content):
    return content.find(class_='mw-page-title-main').text

# Функция поиска всех имен собственных, указанных в статье
def get_proper_nouns(main_text):

    # Создание списка для хранения всех имен собственных
    proper_nouns = []

    # Разбиение основого содержимого страницы на отдельные предложения
    sentences = nltk.sent_tokenize(main_text)

    # Запуск обработки каждого полученного предложения
    for sentence in sentences:

        # Разбиение каждого отдельного предложения на массив слов
        words = nltk.word_tokenize(sentence)

        # Определение, какой частью речи является каждое обрабатываемое слово
        tagged_words = nltk.pos_tag(words)

        # Исключение из массива слов являющихся в предложении первыми, а также имен нарицательных 
        for i, word in enumerate(words):
            is_proper_noun = i != 0 \
                             and word[0].isupper() \
                             and word not in proper_nouns
            if is_proper_noun:
                proper_nouns.append(word)

    # Возвращение массива всех имен собственных
    return proper_nouns

# Функция получения частоты появления в статье названия города
def get_city_freq(city, main_text):

    # Декомпозиция текста на отдельные слова
    words = nltk.word_tokenize(main_text)
    
    # Создание объекта частотного распределения FreqDist
    freq_dist = nltk.FreqDist(words)
    
    # Получение количества упоминаний названия города в тексте
    city_count = freq_dist[city] if city in freq_dist else 0
    
    # Получение общего количества слов в тексте
    total_words = len(words)
    
    # Вычисление частоты появления названия города в тексте в процентном формате
    frequency_percent = (city_count / total_words) * 100 if total_words > 0 else 0
    
    # Возвращение частоты появления названия города в статье в процентном формате
    return "{:.2f}%".format(frequency_percent)

# Создание объекта коммуникатора MPI для всех запущенных процессов
comm = MPI.COMM_WORLD

# Получение номера текущего процесса в коммуникаторе
rank = comm.Get_rank()

# Получение общего количества процессов в коммуникаторе
num_processes = comm.Get_size()

# Отсчет времени
total_t1 = MPI.Wtime()
t1 = time.process_time()

# Вывод номера текущего процесса
print('\nProcess #' + str(rank))

# Получение списка html-страниц 
files = [f for f in glob.glob("html" + "**/*.html", recursive=True)]

# Получение количества файлов, которые нужно обработать
count = int((len(files) / num_processes))

# Получение начального индекса диапазона файлов
start = rank * count

# Получение конечного индекса диапазона файлов
end = start + count

# Создание пустого списка для хранения результатов обработки страниц
processor_cities = []

# Запуск обработки файлов по их индексам
for i in range(start, end):

    # Получение содержания html-страницы и очистка от ненужных блоков
    content = get_text_from_html(files[i])

    # Получение названия города, о котором идет речь в полученной статье
    city = get_city(content)

    # Получение текстового содержимого страницы без бокового меню
    main_text = get_main_text(content)

    # Получение всех имен собственных, указанных в статье
    proper_nouns = get_proper_nouns(main_text)

    # Получение частоты появления в статье названия города
    freq = get_city_freq(city, main_text)

    # Добавление результатов обработки в общий список
    processor_cities.append(CityItem(city, freq, len(proper_nouns)))

# Этот участок кода собирает результаты обработки страниц из всех процессов и передает их в процесс с идентификатором 0

# Получение страницы с самой высокой частотой упоминания своего города
max_freq_cities = comm.gather(max_by_freq(processor_cities))

# Получение страницы с наименьшим количеством собственных имен
min_proper_nouns_cities = comm.gather(min_by_proper_nouns_count(processor_cities))

# Получение страницы с наибольшим количеством собственных имен
max_proper_nouns_cities = comm.gather(max_by_proper_nouns_count(processor_cities))

t2 = time.process_time()
t = t2 - t1

# Сохранение в списке значений времени выполнения каждого процесса
time_list = comm.gather(t)

print('Execute time = ' + str(t))

# Этот блок кода выполняется только в процессе с рангом 0 (главном процессе) и предназначен для вывода результатов анализа страниц
if rank == 0:
    # Получаем окончательные результаты, сравнивая данные из всех процессов
    max_freq = max_by_freq(max_freq_cities)
    max_proper_nouns = max_by_proper_nouns_count(max_proper_nouns_cities)
    min_proper_nouns = min_by_proper_nouns_count(min_proper_nouns_cities)

    total_time = MPI.Wtime() - total_t1

    print('\nСтраница с наивысшей частотой упоминания названия своего города: ' + str(max_freq) + '\n')
    print('Страница с наибольшим количеством собственных имен: ' + str(max_proper_nouns) + '\n')
    print('Страница с наименьшим количеством собственных имен: ' + str(min_proper_nouns) + '\n')
    print('Время выполнения программы составило ' + str(total_time) + ' секунд\n')
