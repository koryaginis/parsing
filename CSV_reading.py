import collections
import itertools
import time
from multiprocessing.pool import ThreadPool
import pandas as pd
from mpi4py import MPI
import csv
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


# Получение наиболее упоминаемых пользователей в твитах
def most_frequent_users(left_pointer, right_pointer, end, data, result_queue):
    # Выборка твитов из набора данных, соответствующих указанным границам
    tweets = data.loc[left_pointer:right_pointer]
    # Подсчет количества упоминаний каждого пользователя
    user_counts = collections.Counter(tweets['UserMentionNames'])
    # Создание словаря для хранения самых часто упоминаемых пользователей и их количества упоминаний
    top_users = {}
    # Перебор пользователей и их количества упоминаний, полученных с помощью Counter
    for t in user_counts.most_common():
        # Запись пользователя и его количества упоминаний в словарь
        top_users[t[0]] = t[1]
    # Помещение словаря с самыми часто упоминаемыми пользователями в очередь результатов
    result_queue.put(top_users)


# Для каждого из пяти пользователей получение десяти наиболее упоминаемых хештегов
# Функция принимает набор твитов и словарь с пятью наиболее упоминаемыми пользователями и их количеством упоминаний
def find_popular_hashtags(tweets, top_users):
    # Словарь для хранения наиболее популярных хештегов для каждого из пяти пользователей
    top_hashtags = {}
    # Перебор пяти наиболее упоминаемых пользователей
    for user in top_users:
        # Фильтрация твитов по конкретному пользователю
        user_tweets = tweets[tweets['UserMentionNames'] == user[0]].copy()
        # Заполнение отсутствующих значений в колонке хештегов пустой строкой
        user_tweets['Hashtags'] = user_tweets['Hashtags'].fillna('')
        # Подсчет количества упоминаний каждого хештега для данного пользователя
        hashtag_counts = collections.Counter([tag for tags in user_tweets['Hashtags'] for tag in tags.split() if tag])
        # Выборка десяти наиболее упоминаемых хештегов для данного пользователя
        top_hashtags[user[0]] = [t[0] for t in hashtag_counts.most_common()[:10]]
    return top_hashtags


# Выбор пользователя с наиболее отличающимся списком хештегов
# Функция принимает словарь с пользователями и их списками хештегов
def find_distinct_user(top_hashtags):
    # Создание словаря для хранения множеств уникальных хештегов для каждого пользователя
    distinct_hashtags = {}
    # Преобразование списков хештегов в множества для каждого пользователя
    for user, hashtags in top_hashtags.items():
        distinct_hashtags[user] = set(hashtags)
    # Создание словаря для хранения количества уникальных хештегов для каждого пользователя    
    distinct_counts = {}
    # Подсчет количества уникальных хештегов для каждого пользователя
    for user, hashtags in distinct_hashtags.items():
        distinct_counts[user] = len(hashtags)
    # Возвращение пользователя с наибольшим количеством уникальных хештегов
    return max(distinct_counts, key=distinct_counts.get)


# Выбор двух пользователей с наиболее похожими списками хештегов
# Функция принимает список пяти самых упоминаемых пользователей и словарь с хештегами для каждого пользователя
def similiar_users_by_hashtag(top_users, top_hashtags):
    # Создаем список для хранения имен пользователей из списка top_users
    top_users_names = []
    # Проходимся по каждому пользователю в списке top_users
    for user in top_users:
        # Добавляем имя пользователя в список top_users_names
        top_users_names.append(user[0])
    # Выводим на печать пять самых упоминаемых пользователей
    print("\nПять самых упоминаемых пользователей: ", top_users_names)
    # Создаем словарь для хранения коэффициентов схожести между пользователями
    similarity_scores = {}
    # Проходимся по всем парам пользователей из списка top_users_names
    for pair in itertools.combinations(top_users_names, 2):
        # Находим пересечение хештегов у двух пользователей
        intersection = set(top_hashtags[pair[0]]).intersection(set(top_hashtags[pair[1]]))
        # Записываем размер пересечения в качестве коэффициента схожести
        similarity_scores[pair] = len(intersection)
    # Возвращаем пару пользователей с наибольшим коэффициентом схожести
    return max(similarity_scores, key=similarity_scores.get)


# Объединение словарей, содержащих пользователей и подсчет упоминаний этих пользова-телей
def merge_dicts(dictionaries):
    # Создаем пустой словарь для хранения результата объединения
    result = {}
    # Проходимся по каждому словарю в списке dictionaries
    for dictionary in dictionaries:
        # Проходимся по объединенному множеству ключей из текущего словаря и результата
        for k in set(result) | set(dictionary):
            # Для каждого ключа обновляем значение в результирующем словаре,
            # прибавляя значение из текущего словаря или 0, если ключ отсутствует
            result[k] = result.get(k, 0) + dictionary.get(k, 0)
    # Возвращаем объединенный словарь
    return result


# Создание коммуникатора MPI
comm = MPI.COMM_WORLD
# Получение номера текущего процесса
rank = comm.Get_rank()
# Получение общего количества процессов
num_processes = comm.Get_size()


# Отсчет времени
t1 = MPI.Wtime()


# Чтение файла с набором данных
data = pd.read_csv('csv/FIFA.csv')


# Получение количества строк для обработки отдельным процессом
count = int((len(data) / num_processes))
# Получение подмножества строк для обработки отдельным процессором
start = rank * count
end = start + count
process_data = data.loc[start:end]


# Распределение данных между потоками
threads = 1
count_on_thread = count / threads # Количество строк на поток
start_by_thread = start
end_by_thread = start_by_thread + count_on_thread


# Создание массива данных для использования каждым потоком
data_on_thread = []
while start_by_thread < end:
    data_on_thread.append((start_by_thread, end_by_thread, end, process_data))
    start_by_thread += count_on_thread
    end_by_thread += count_on_thread


result_queue = Queue()


# Создаем пустой массив для хранения результатов
thread_results = []


# Запускаем потоки
with ThreadPoolExecutor(max_workers = threads) as pool:
    for i in range(threads):
        pool.submit(most_frequent_users, data_on_thread[i][0], data_on_thread[i][1], data_on_thread[i][2], data_on_thread[i][3], result_queue)

while not result_queue.empty():
    result = result_queue.get()
    thread_results.append(result)

# Сбор результатов с разных процессов
local_result = comm.gather(merge_dicts(thread_results))


# Получаем окончательные результаты, получая и сравнивая данные из всех процессов
if rank == 0:
    # объединение словарей результатов, полученных с разных процессов, в единый словарь.
    # Сортировка объединенного словаря по количеству упоминаний пользователей в убывающем порядке и 
    # Ограничение его пятью самыми часто упоминаемыми пользователями
    five_most_frequent_users = sorted(merge_dicts(local_result).items(), key=lambda item: item[1],
                                      reverse=True)[:5]
    # Поиск десяти наиболее популярных хештегов для каждого из пяти самых часто упоминаемых пользователей
    top_hashtags = find_popular_hashtags(data, five_most_frequent_users)
    # Определение пользователя с наиболее различным списком хештегов
    most_distinct_user = find_distinct_user(top_hashtags)
    # Поиск пары пользователей с наиболее похожими списками хештегов среди пяти самых часто упоминаемых пользователей
    most_similar_pair = similiar_users_by_hashtag(five_most_frequent_users, top_hashtags)


    print('\nПользователь, у которого 10 хештегов наиболее отличаются от четырех остальных пользователей: ', most_distinct_user)
    print('\nДва пользователя, у которых распределения частоты встречаемости 10 хеште-гов наиболее близки/похожи друг на друга: ', most_similar_pair)


    t2 = MPI.Wtime()
    t = t2 - t1


    print('\nTotal time:', t, '\n')