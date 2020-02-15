#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import numpy as np
import matplotlib.pyplot as plt
import math
from visualization_msgs.msg import *
from geometry_msgs.msg import TwistStamped, Twist, Pose, Quaternion, Point
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt


def get_max_height(data, pose):
    """
    вычисляем расстояние до потолка с учетом наклона
    :type data: LaserScan
    :type angle: Pose
    :param data: данные сонара
    :param angle: углы эйлера в радианах
    :return: растсояние
    """
    dist_lidar = np.min(data)
    height =  (math.cos(pose.orientation.x)*dist_lidar)*math.cos(pose.orientation.y)
    return height + pose.position.z


def sonar_to_sectors_direct(sonar_data, orientation, max_dist_lidar=34):
    """"
    Делим данные лидара на 4 сектора. 0 (по часовой) задние данные
    :type orientation: Quaternion
    :return массив секторов с препятствиями
    """
    # if abs(orientation.x) > math.radians(5) or abs(orientation.y) > math.radians(5):
    #     return None

    range_of_sectors = 4
    # поворачиваем данные вдоль X
    summAngle = orientation.z + (-math.pi / 4)
    srez = int(len(sonar_data) / (2*math.pi) * summAngle)
    sonar_data = sonar_data[srez:] + sonar_data[:srez]
    # избавляемся от inf
    without_inf = list()
    for k in range(0, len(sonar_data)):
        if np.isinf(sonar_data[k]):
            without_inf.append(max_dist_lidar)
        else:
            without_inf.append(sonar_data[k])
    # инвертируем сонар без inf
    sonar_data = without_inf

    # количество значений в секторе
    samples_of_sector = len(sonar_data) / range_of_sectors

    sectors = list()

    for i in range(0, range_of_sectors):
        data_of_sector = np.asarray(sonar_data[samples_of_sector * i:samples_of_sector * (i + 1)])
        data_sector = np.median(data_of_sector)
        data_sector = max_dist_lidar if math.isinf(data_sector) else data_sector
        sectors.append(data_sector)
    return sectors


def get_pos_to_center(walls, current_pose):
    """
    получаем координаты до центра комнаты
    :type walls: []
    :type current_pose: Pose
    :param walls:
    :param current_pose:
    :return: x, y
    """
    x = current_pose.position.x - ((walls[0] + walls[2])/2.0 - walls[2])
    y = current_pose.position.y + (walls[1] - ((walls[1] + walls[3]) / 2.0))
    return x, y


def is_center_of_room(walls):
    """
    проверяем, находимся ли мы в центре комнаты

    :param walls:
    :return:
    """
    if not walls:
        return None

    data1 = walls[0] - walls[2]
    data2 = walls[1] - walls[3]

    data = np.mean([abs(data1), abs(data2)])

    # Вычисляем область середины комнаты. Она равна 10% от общей длины
    center1 = (walls[0] + walls[2]) / 10.0
    center2 = (walls[1] + walls[3]) / 10.0
    center = np.mean([center1, center2])

    if data < center:
        return True, center
    else:
        return False, center
