from __future__ import division
from __future__ import print_function
import telebot
from telebot import *
from requests import get
from random import randint
import glob
import io
from PIL import Image
import cv2 as cv

import numpy as np


bot = telebot.TeleBot('API_TOKEN')


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Salad and Boui bot v1")
        # Готовим кнопки
        keyboard = types.InlineKeyboardMarkup()
        # По очереди готовим текст и обработчик для каждого знака зодиака
        key_about_todo = types.InlineKeyboardButton(text='What can this bot do?', callback_data='about_todo')
        # И добавляем кнопку на экран
        keyboard.add(key_about_todo)
        key_about_author = types.InlineKeyboardButton(text='About author', callback_data='about_author')
        keyboard.add(key_about_author)
        key_todo = types.InlineKeyboardButton(text='Send photo', callback_data='todo')
        keyboard.add(key_todo)
        # Показываем все кнопки сразу и пишем сообщение о выборе
        bot.send_message(message.from_user.id, text='Choose what do you want to know/do: ', reply_markup=keyboard)
    elif message.text == "/help":
        bot.send_message(message.from_group.id, "Type /start")
    else:
        bot.send_message(message.from_group.id, "I don't understand you. Type /help.")

    if message.text == "cat" or message.text == "Cat":
        bot.send_photo(message.from_user.id, photo=open('cat.jpg', 'rb'))


@bot.message_handler(content_types=['photo'])
def get_photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    open(str(fileID) + ".jpg", 'wb').write(downloaded_file)

    def masked(frame):
        """
        Выдает обработанное изображение с учетом фона
        """
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_blue = np.array([0, 0, 120])
        upper_blue = np.array([180, 38, 255])
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        result = cv.bitwise_and(frame, frame, mask=mask)
        b, g, r = cv.split(result)
        flt = g.copy()
        ret, mask = cv.threshold(flt, 10, 255, 1)
        frame[mask == 0] = 255
        return frame

    def similarness(img1, img2) -> float:
        """
        Выдает коэффициент похожести между двумя изображениями
        """
        # Меняем цветовое пространство на HSV format
        hsv_base = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
        hsv_test1 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

        # Инициализация переменных для составления гистограммы
        h_bins = 50
        s_bins = 60
        histSize = [h_bins, s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges  # размер наших цветовых срезов

        # Используем нулевой и первый канал
        channels = [0, 1]

        # Создание гистограмм и их нормализация
        hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
        cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
        cv.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        # Нуль в качестве последнего элемента означает сравнение по методу корреляции
        return cv.compareHist(hist_base, hist_test1, 0)

    # Указываем путь к папке с салатницами + /*.jpg
    path_to_salad_bowl = '/Users/mihailmurunov/PycharmProjects/SaladBowlBot/salad-dish/*.jpg'
    images = glob.glob(path_to_salad_bowl)
    person = cv.imdecode(np.frombuffer(downloaded_file, np.uint8), -1)
    # Прогоняем для каждого человека с каждым изображением салатницы функцию похожести
    answers = [0, 0]
    bot.send_message(message.from_user.id, "Wait 10 seconds please...")
    for image in images:
        tmp = similarness(masked(cv.resize(person, (200, 200))),
                            masked(cv.resize(cv.imread(image), (200, 200))))
        if answers[0] < tmp:
            answers[0] = tmp
            answers[1] = image
    # Открываем это изображение и показываем
    bot.send_message(message.from_user.id, "Look at result!")
    bot.send_photo(message.from_user.id, photo=open(answers[1], 'rb'))

@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    # Если нажали на одну из 3 кнопок — выводим соотствующее
    if call.data == "about_todo":
        # Формируем сообщение
        msg = "I can choose the best salad bowl by your photo!"
        # Отправляем текст в Телеграм
        bot.send_message(call.message.chat.id, msg)
    if call.data == "about_author":
        # Формируем сообщение
        msg = "This bot was made by Mikhail Murunov"
        # Отправляем текст в Телеграм
        bot.send_message(call.message.chat.id, msg)
    if call.data == "todo":
        # Формируем сообщение
        msg = "Waiting for photo..."
        # Отправляем текст в Телеграм
        bot.send_message(call.message.chat.id, msg)


bot.polling(none_stop=True, interval=0)

