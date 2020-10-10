from __future__ import division
from __future__ import print_function

import glob

import cv2 as cv

import numpy as np


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


# Указываем путь к папке с людьми + /*.jpg
path_to_person = '/Users/mihailmurunov/PycharmProjects/SaladBowlandBoui/person/*.jpg'

# Указываем путь к папке с салатницами + /*.jpg
path_to_salad_bowl = '/Users/mihailmurunov/PycharmProjects/SaladBowlandBoui/salad-dish/*.jpg'

persons = glob.glob(path_to_person)
sorted(persons)
images = glob.glob(path_to_salad_bowl)
sorted(images)

# Прогоняем для каждого человека с каждым изображением салатницы функцию похожести
for person in persons:
    answers = [0, 0]
    for image in images:
        tmp = similarness(masked(cv.resize(cv.imread(person), (200, 200))),
                          masked(cv.resize(cv.imread(image), (200, 200))))
        if answers[0] < tmp:
            answers[0] = tmp
            answers[1] = image
    # Открываем это изображение и показываем
    guess = cv.imread(answers[1])
    cur_person = cv.imread(person)
    cv.imshow("Person", cur_person)
    cv.imshow("Guess", guess)
    cv.waitKey(0)
    cv.destroyAllWindows()
