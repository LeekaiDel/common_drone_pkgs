#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math

import rospy
import tf
from visualization_msgs.msg import *
from geometry_msgs.msg import TwistStamped, Twist
from sensor_msgs.msg import LaserScan
from drone_msgs.msg import WindowPointDir

import detector_tools as detector

sonar_data = list()
window_pose = WindowPointDir()
current_course = [0.0, 0.0,0.0]
current_pose = [0.0,0.0,0.0]

pub_marker = rospy.Publisher("/marker_window", Marker,queue_size=10)
pub_window = rospy.Publisher("/window_detector/point_dir", WindowPointDir, queue_size=10)

def setup_market(x,y,z,found):
    """
    :type found: bool
    :param pose:
    :return:
    """
    marker = Marker()
    marker.header.frame_id = "/base_footprint"
    marker.header.stamp = rospy.get_rostime()
    marker.ns = "sonar_window_marker"
    marker.id = 0
    marker.action = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.type = Marker.CUBE
    marker.color.r = 1.0 if not found else 0.0
    marker.color.g = 1.0 if found else 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z

    return marker

def callbackSonar(data):
    """
    :type data: LaserScan
    :param data:
    :return:
    """
    global sonar_data, current_course
    sonar_data = detector.sonar_around_course(data.ranges, current_course[2])


def get_current_course_and_height(listener):
    """
    Получаем текущие координаты дроона
    :param listener:
    :return:
    """
    global current_course, height
    try:
        (trans, rot) = listener.lookupTransform('base_footprint', 'base_link', rospy.Time(0))
        current_pose[0] = trans[0]
        current_pose[1] = trans[1]
        current_pose[2] = trans[2]
        # Получаем текущий курс дрона
        current_course = tf.transformations.euler_from_quaternion(rot)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("ERROR")

def rotate_vect(rot,dist):
    """
    Поворачиваем на заданный угол относительно a на угол rot
    :param rot: угол порота
    :return: возвращаем точку повёрнутую на нужный угол

    """
    rotate = np.array([[math.cos(rot), -math.sin(rot)],
                       [math.sin(rot), math.cos(rot)]])

    pos = np.array([[dist],[0.0]])
    val = np.dot(rotate,pos)

    return val


if __name__ == '__main__':
    """
       Основной цикл узла ROS.

       :return: код завершения программы
       """

    rospy.init_node('find_window_sonar_node', anonymous=True)
    rate = rospy.Rate(10)

    # подписываемся на считываение сонара
    rospy.Subscriber("/rplidar/scan", LaserScan, callbackSonar)
    listener = tf.TransformListener()
    xy_pos = [-100.0, -100.0]
    while (not rospy.is_shutdown()):
        #
        # # если пришли данные с сонара ищем окно
        get_current_course_and_height(listener)
        if sonar_data:
            infZone = detector.find_borders(sonar_data)
            window_index = None

            try:
                window_index = detector.find_window(sonar_data, infZone)
            except:
                print("error")
                window_index = None

            if window_index: # and math.degrees(abs(current_course[1])) < 10 and math.degrees(abs(current_course[2])) < 10:
                val = detector.get_val(sonar_data, window_index)
                # print("find:",val, window_index)
                # публикуем в tf относительно дрона
                br = tf.TransformBroadcaster()

                xy_pos = rotate_vect(val[0],val[1])

                window_pose.point.course = val[0]
                print ("angle sonar", math.degrees(val[0]))
                    #if val[0] > 0 else (2 * math.pi) + val[0]

                window_pose.point.point.x = xy_pos[0]
                window_pose.point.point.y = xy_pos[1]
                window_pose.point.point.z = current_pose[2]
                window_pose.found_window = True

                pub_window.publish(window_pose)
                pub_marker.publish(setup_market(xy_pos[0],
                                                xy_pos[1],
                                                current_pose[2], True))
            else:
                # print("window not found")
                window_pose.found_window = False

                pub_window.publish(window_pose)
                pub_marker.publish(setup_market(xy_pos[0],
                                                xy_pos[1],
                                                current_pose[2], False))
        rate.sleep()
