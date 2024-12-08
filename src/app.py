import telebot
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

def download_db():
    # URL вашей таблицы с gid листа
    SPREADSHEET_URL = os.getenv("DB_URL")  # Получаем URL базы данных из переменной окружения

    # Имя файла для сохранения данных
    OUTPUT_FILE = os.getenv("DB_FILE")  # Получаем путь к файлу базы данных

    try:
        # Попробуем загрузить данные напрямую в DataFrame
        df = pd.read_csv(SPREADSHEET_URL)

        # Сохраняем новые данные в файл
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8", sep=",")
        logger.info("Данные успешно экспортированы в %s", OUTPUT_FILE)
    except Exception as e:
        logger.error("Ошибка при загрузке данных: %s", e)

        # Если файл уже существует, оставляем его содержимое
        if os.path.exists(OUTPUT_FILE):
            logger.info("Используем существующий файл %s.", OUTPUT_FILE)
        else:
            logger.info("Файл данных отсутствует, и загрузка не удалась. Проверьте источник данных.")


# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка базы данных
download_db()

# Загрузка данных из локального CSV файла
try:
    logger.info("Загрузка данных из файла %s", os.getenv("DB_FILE"))  # Чтение из переменной окружения
    movie_dataset = pd.read_csv(os.getenv("DB_FILE"))  # Чтение из переменной окружения
    logger.info("Данные успешно загружены")
except Exception as load_error:
    logger.error("Ошибка при загрузке данных: %s", load_error)
    raise

# Преобразование жанров к строковому формату
movie_dataset['genre'] = movie_dataset['genre'].astype('str')
logger.info("Жанры преобразованы в строковый формат")

# Оставляем только один жанр в колонке 'genre'
movie_dataset['genre'] = movie_dataset['genre'].apply(lambda genre_list: genre_list.split(',')[0])
logger.info("Оставлен только один жанр для каждого фильма")

# Инициализация Telegram-бота
try:
    movie_bot = telebot.TeleBot(os.getenv("BOT_TOKEN"))  # Получаем токен бота из переменной окружения
    logger.info("Бот успешно инициализирован")
except Exception as bot_init_error:
    logger.error("Ошибка инициализации бота: %s", bot_init_error)
    raise

# Функция для поиска похожих фильмов
def find_similar_movies(movie_data, user_query_description, max_recommendations=5):
    logger.info("Начат поиск %d похожих фильмов", max_recommendations)
    try:
        tfidf_vectorizer = TfidfVectorizer()  # Создаем векторизатор TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform(movie_data['description'])  # Матрица TF-IDF для описаний фильмов
        query_tfidf_vector = tfidf_vectorizer.transform([user_query_description])  # Преобразуем запрос пользователя в TF-IDF
        similarity_scores = cosine_similarity(query_tfidf_vector, tfidf_matrix).flatten()  # Косинусное сходство
        top_movie_indices = similarity_scores.argsort()[-1:-max_recommendations - 1:-1]  # Индексы похожих фильмов
        logger.info("Поиск завершен, найдено %d фильмов", len(top_movie_indices))
        return movie_data.iloc[top_movie_indices]
    except Exception as search_error:
        logger.error("Ошибка при поиске похожих фильмов: %s", search_error)
        return pd.DataFrame()  # Возвращаем пустой DataFrame при ошибке

# Обработчик текстовых сообщений от пользователя
@movie_bot.message_handler(content_types=['text'])
def handle_user_query(message):
    user_description_query = message.text
    logger.info("Получен запрос от пользователя: %s", user_description_query)
    try:
        similar_movies = find_similar_movies(movie_dataset, user_description_query)
        if similar_movies.empty:
            movie_bot.send_message(message.from_user.id, "Извините, похожих фильмов не найдено.")
            logger.info("Похожих фильмов не найдено для запроса: %s", user_description_query)
            return

        similar_movies_dict = similar_movies.to_dict()
        movie_titles = similar_movies_dict['title']
        movie_years = similar_movies_dict['year']
        movie_descriptions = similar_movies_dict['description']
        response_movies = [
            {"title": movie_titles[index], "year": movie_years[index], "description": movie_descriptions[index]} 
            for index in movie_titles
        ]
        response_text = "\n\n".join(
            f"🎬 {movie['title']} ({movie['year']})\n{movie['description']}\n---------------"
            for movie in response_movies
        )
        movie_bot.send_message(message.from_user.id, f"Похожие фильмы:\n{response_text}")
        logger.info("Результаты успешно отправлены пользователю")
    except Exception as handler_error:
        logger.error("Ошибка при обработке запроса: %s", handler_error)
        movie_bot.send_message(message.from_user.id, "Произошла ошибка. Попробуйте позже.")

# Запуск бота
try:
    logger.info("Бот запущен, ожидается ввод сообщений")
    movie_bot.polling(none_stop=True, interval=0)
except Exception as polling_error:
    logger.error("Ошибка при запуске бота: %s", polling_error)

