#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import copy

_borders = []

def find_borders(lidar_data):
    """
    Возвращаем массив из границ интервалов с inf [слева от inf, справа от inf]
    :type lidar_data: []
    :type _borders: list
    :param lidar_data: данные лидара
    :return: [,]
    """
    global _borders
    del _borders[:]

    _left = None
    _right = None
    #перебираем массив лидара и ищем inf

    # если первый элемент inf, то смотрим с конца
    if math.isinf(lidar_data[0]):
        for k in range(len(lidar_data) - 1, -1, -1):
            if math.isinf(lidar_data[k]) and (not math.isinf(lidar_data[k - 1])):
                _left = k - 1
                break
    for i in range(1,len(lidar_data)-1):

        if np.isinf(lidar_data[i]):
            if (_left == None) and (not math.isinf(lidar_data[i-1])):
                _left = i - 1

            if (_right == None) and (not math.isinf(lidar_data[i+1])):
                _right = i + 1

        if _left != None and _right != None:
            _borders.append([_left,_right])
            _left = None
            _right = None
    # print("_borders:", len(_borders))
    return _borders

def is_this_line(points_x,points_y):
    """
    Функция проверки лежат ли ТРИ точки на одной прямой
    :param points_x:
    :param points_y:
    :return:
    """
    # 1. Проверяем параллельность осям координат
    dist = 0.003

    if abs(points_x[0]-points_x[1])<dist and abs(points_x[1]-points_x[2])<dist:

        return True;

    if abs(points_y[0]-points_y[1])<dist and abs(points_y[1]-points_y[2])<dist:

        return True;

    # 2. Если не параллельны осям
    tmp1=(points_x[2]-points_x[0])/(points_x[1]-points_x[0]);
    tmp2=(points_y[2]-points_y[0])/(points_y[1]-points_y[0]);
    if abs(tmp1-tmp2)<0.001:

        return True;
    else:
        return False;

def find_window(D_lidar, borders):
    """
        Функция поиска окна (анализируем границы интервала с Inf)
        :type D_lidar: []
        :type borders: []
        :param D_lidar: данные с сонара
        :param borders: границы выбросов
        :param window: массив лучей мужду которыми окно
        :return:
        """
    ray_range = 33.5
    window = []
    samles_angle = math.pi * 2 / len(D_lidar)
    del window[:]
    # Кидаем None на всякий случай, вдруг окна в поле зрения нет
    # 1-фаза. Ищем расстояния в ряду данных между тенями. Самые большие расстояния
    #дадут ДВА потенциальных окна или ОДНО потенциальное окно если область тени изначально одна

    if borders:
        d1 = []
        del d1[:]
        if len(borders) > 1:
            d1 = []
            for i in range(1,len(borders)):
                d1.append(abs(borders[i][0] - borders[i - 1][1]))  # Ищем расстояние между тенями

            # Так как лидар круговой - замыкаем его
            d1.append(abs(borders[0][0] - borders[-1][1]))
            # tmp_min = np.nonzero(d1 == np.min(d1))
            # if np.min(d1) < 10: # если столб между окна
            #     index = tmp_min[0][0]
            #     borders[index][1] = borders[index+1][1]
            #     del borders[index+1]

            if len(d1) == 1:
                window = borders[0]

            elif len(d1) == 2:  # Если всего две тени, то обе добавляем в потенциальные окна
                ind = [0,1]
            else:  # Если больше двух теней, то добавляем 2 тени с максимальным расстоянием между ними
                # в потенциальные окна

                tmp_array = np.nonzero(d1 == np.max(d1))  # Ищем максимальное расстояние между тенями
                tmp = tmp_array[0][0]
                if tmp == len(borders)-1:
                    ind = [0, tmp]  # В случае если это растояние было после последней тени в массива, замыкаем его на первую тень
                    window.append(borders[ind[1]]) # Формируем потенциальные окна - координаты конца
                    window.append(borders[ind[0]]) # Формируем потенциальные окна - координаты начала
                elif tmp != 0:
                    ind = [tmp, tmp+1]  # Если максимальное расстояние было в середине массива, замыкаем его на предыдущую тень
                    window.append(borders[ind[1]]) # Формируем потенциальные окна - координаты конца
                    window.append(borders[ind[0]]) # Формируем потенциальные окна - координаты начала
                elif tmp == 0:
                    ind = [tmp]  # Если тень была одна, добавляем ее в потенциальное окно
                    window.append(borders[tmp]) # Формируем потенциальные окна - координаты конца
        else:
            window = borders[0]  # Формируем потенциальное окно, если область тени изначально одна - координаты начала
    # print ("len window: ", len(window))
    # print ("borders: ", borders)

    for i in range(len(window)-1, 0-1):
        val1 = D_lidar[window[i][0]]
        val2 = D_lidar[window[i][1]]
        if val1 > ray_range or val2 > ray_range:
            del window[i]
    # окно - может быть настоящим
    if window:  # Если потенциальные окна есть
        if len(window) > 1:  # Если оно не одно
            d1 = D_lidar[window[0][0]]
            d2 = D_lidar[window[0][1]]
            d3 = D_lidar[window[1][0]]
            d4 = D_lidar[window[1][1]]
            tmp1 = np.max([d1,d2])  # Ищем максимальную дальность от дрона до первого окна
            tmp2 = np.max([d3,d4])  # Ищем максимальную дальность от дрона до второго окна

            if (tmp1 < tmp2):  # Если первое окно ближе второго  - оно может быть настоящим
                del window[1]
                print ("test 1")
            elif (tmp2 < tmp1):  # Если второе окно ближе первого - оно может быть настоящим
                del window[0]
                print ("test 2")

    else:  # Если расстояния были большие - это не окна, а тени
        del window[:]

    ## 3-фаза. Проверяем лежат ли справа или слева от потенциального окна 3 точки на одной прямой.
    # Если лежат, значит это окно.
    if window:
        # Избавляемся от вложенных списков
        window = merge(window)
        test_left = False
        test_right = False
        offer = 3
        print (window[0])
        if window[0] > 3+offer:
            points_x = [D_lidar[window[0]-offer] * math.cos(0),
                        D_lidar[window[0] - 1-offer] * math.cos(-samles_angle),
                        D_lidar[window[0] - 2-offer] * math.cos(-2 * samles_angle)]

            points_y = [D_lidar[window[0]-offer] * math.sin(0),
                        D_lidar[window[0] - 1-offer] * math.sin(-samles_angle),
                        D_lidar[window[0] - 2-offer] * math.sin(-2 * samles_angle)]

            test_left = is_this_line(points_x, points_y)
        if window[1] < len(D_lidar) - 3-offer:
            points_x = [D_lidar[window[1]+offer] * math.cos(0),
                        D_lidar[window[1] + 1+offer] * math.cos(samles_angle),
                        D_lidar[window[1] + 2+offer] * math.cos(2 * samles_angle)]
            points_y = [D_lidar[window[1]+offer] * math.sin(0),
                        D_lidar[window[1] + 2+offer] * math.sin(samles_angle),
                        D_lidar[window[1] + 3+offer] * math.sin(2 * samles_angle)]



            test_right = is_this_line(points_x, points_y)
        if (not test_left) and (not test_right):
            del window[:]
    val = copy.copy(window)

    del window[:]
    if val:
        return [val[0],val[1]]
    else:
        return None

def get_val(sonar_data,index, offser = -math.pi, max_dist = 34):
    """
    Получаем значение угла относительно курса, а также расстоние
    :param sonar_data:
    :param index:
    :param offser:
    :param lidar_cout:
    :return: [angle, dist]
    """
    lidar_cout = len(sonar_data)
    angle = 0
    dist = 0.0

    for i in range(0,len(index)):
        angle = angle + offser + (math.pi*2 / lidar_cout * index[i])
        dist = dist + sonar_data[index[i]]

    angle /= len(index)
    dist /= len(index)
    dist = dist if not math.isinf(dist) else max_dist
    return -angle, dist

def merge(lst, res=[]):
  for el in lst:
    merge(el) if isinstance(el, list) else res.append(el)
  return res

def sonar_around_course(sonar_data, course):
    # поворачиваем данные относительно куда целевой точки
    srez = int(len(sonar_data) / (2 * math.pi) * course)
    return sonar_data[srez:] + sonar_data[:srez]