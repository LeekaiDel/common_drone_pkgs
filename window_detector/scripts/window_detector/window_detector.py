#!/usr/bin/env python
# coding=utf8

# window_detector.py
#
# Описание
# Узел ROS для визуального детектирования окна
# Использует OpenCV для детектирования замкнутых контуров и данные лидаров.
# Окном считается замкнутый контур.
#
# Входные данные:
#   Изображение
#     Топик: /stereo/left/image_raw
#     Тип: sensor_msgs.msg.Image
#   8-ми секционный лидар:
#     Топик: /lidar_vu8_front/sectored
#     Тип: sensor_msgs.msg.LaserScan
#   Круговой лидар:
#     Топик: /rplidar/scan
#     Тип: sensor_msgs.msg.LaserScan
#
# Выходные данные:
#   Изображение с результатом детектирования
#     Топик: /window_detector/image_raw
#     Тип: sensor_msgs.msg.Image
#   Углы отклонения центра изображения от центра окна
#     Топик: /window_detector/angle_dir
#     Тип: drone_msgs.msg.WindowAngleDir
#

import rospy
import math
import numpy as np
from numpy import degrees
import cv2
import roslib
from cv_bridge import *
from copy import deepcopy
# from numpy.distutils.system_info import lapack_src_info
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from drone_msgs.msg import WindowAngleDir
from stereo_msgs.msg import DisparityImage
# import tf.transformations as t


def is_cv2():
    """
    Проверка 2 версии OpenCV.

    :return: 2-я версия?
    :rtype: bool
    """
    major = cv2.__version__.split(".")
    return major[0] == '2'


def is_cv3():
    """
    Проверка 3 версии OpenCV.

    :return: 3-я версия?
    :rtype: bool
    """
    major = cv2.__version__.split(".")
    return major[0] == '3'


print ('cv2 = %s' % is_cv2())
print ('cv3 = %s' % is_cv3())


def get_binary_img_mask(width, height):
    """
    Функция возвращает бинарную маску для отсечения слепых зон карты глубины.

    :param width: ширина изображения
    :type width: int
    :param height: высота изображения
    :type height: int
    :return: бинарная маска
    :rtype: numpy.ndarray
    """
    img = np.ndarray(shape=(height, width), dtype=np.uint8)
    for j in range(width):
        for i in range(height):
            img[i][j] = 0
            # if 40 < j < width - 8 and 8 < i < height - 8:
            if 100 < j < width - 30 and 30 < i < height - 30:
                img[i][j] = 1
    return img


class Lidar:
    """
    Структура описывающая лидар.
    """

    def __init__(self, width=0, height=0, sectors_count=0, min_range=0, max_range=0, name='', data=LaserScan()):
        """
        Конструктор.

        :param width: ширина горизонтальной развертки лидара в радианах
        :type width: float
        :param height: ширина вертикальной развертки лидара в радианах
        :type height: float
        :param sectors_count: количество секотроов лидара
        :type sectors_count: int
        :param min_range: минимальное расстояние
        :type min_range: float
        :param max_range: максимальное расстояние
        :type max_range: float
        :param name: название лидара
        :type name: str
        :param data: данные лидара
        :type data: LaserScan
        """
        self.width = width
        self.height = height
        self.sectors_count = sectors_count
        self.min_range = min_range
        self.max_range = max_range
        self.name = name
        self.data = None

    def __str__(self):
        return "(%s: w = %s; h = %s; sectors_count = %s; min = %s; max = %s)" % \
               (self.name, degrees(self.width), degrees(self.height),
                self.sectors_count, self.min_range, self.max_range)

    def __repr__(self):
        self.__str__()


# --- Global variables
# Лидар с 8-ю секторами
lidar_vu8 = Lidar()
lidar_vu8.name = 'lidar_vu8'
lidar_vu8.width = 0.349066  # ~20 deg
lidar_vu8.height = 0.0523598  # ~3 deg
lidar_vu8.sectors_count = 8
lidar_vu8.min_range = 0.05
lidar_vu8.max_range = 34.0
# Круговой лидар
lidar_360deg = Lidar()
lidar_360deg.name = 'lidar_360deg'
lidar_360deg.width = math.pi*2  # 360 deg
lidar_360deg.sectors_count = 720
lidar_360deg.min_range = 0.4
lidar_360deg.min_range = 34
# Изображение с камеры
image_width_px = 648
image_height_px = 486
image_width_rad = 0.785398  # ~45 deg
image_height_raf = image_height_px * image_width_rad / image_width_px  # ~33 deg

cv_image_ = None  # Изображение в формате OpenCV
res_image_ = None  # Иотговое изображение
binary_image = None  # Бинарное изображение
depth_image = None  # Изображение глубины

window_center_direction = WindowAngleDir()
window_center_direction.found_window = False

masked_img = get_binary_img_mask(image_width_px, image_height_px)

debug_prints = False


def cfg_callback(config, level):
    # rospy.loginfo("""Reconfigure Request: {detect_range}, {cv_threshold},""".format(**config))
    return config


def image_callback(data):
    """
    Camera image callback.

    :param data: msg with image
    :type data: sensor_msgs.msg.Image
    """
    global cv_image_
    try:
        bridge = CvBridge()
        cv_image_ = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        if debug_prints:
            print(e)


def stereo_depth_callback(data):
    """
    Streo image callback.

    :param data: stereo image
    :type data: DisparityImage
    """
    global depth_image
    try:
        bridge = CvBridge()
        depth_image = cv2.bitwise_not(np.uint8(bridge.imgmsg_to_cv2(data.image, "32FC1"))*255) #* masked_img
    except CvBridgeError as e:
        if debug_prints:
            print(e)
        depth_image = None


def point_cloud_callback(data):
    """
    Point cloud subscribe.

    :param data: msg with point cloud data
    :type data: sensor_msgs.msg.PointCloud2
    """
    pass
    # if debug_prints:
    #     print("cloud: %s x %s" % (data.width, data.height))


def lidar_vu8_callback(data):
    """
    Lidar 8 sectors subscribe.

    :param data: laser scan data
    :type data: sensor_msgs.msg.LaserScan
    """
    global lidar_vu8
    lidar_vu8.data = data


def lidar_360deg_callback(data):
    """
    Lidar with 360 deg width subscribe.

    :param data: laser scan data
    :type data: sensor_msgs.msg.LaserScan
    """
    global lidar_360deg
    lidar_360deg.data = data


def compare_with_depth(bin_img):
    """
    Функция возвращает изображение с общими объектами двух изображений.
    Нефига не стабильная функция.

    :param bin_img: бинарное изображения
    :type bin_img: :cpp:type:`cv::Mat`
    :return:
    :rtype: bool
    """
    global depth_image
    if depth_image is None:
        return bin_img
    depth_img = deepcopy(depth_image)
    # depth_img = cv2.bilateralFilter(depth_img, 9, 75, 75)
    depth_img = cv2.bilateralFilter(depth_img, 10, 400, 400)
    _retval, bin_depth = cv2.threshold(depth_img, 150, 255, cv2.THRESH_BINARY)
    res = bin_depth + cv2.bitwise_not(bin_img)
    res = cv2.bitwise_not(res) * masked_img

    return res
  
  
def compare_with_grey_img(cx, cy, cnt, img):
    """
    Функция сравнения области изображения с серым изображением.
    Вернет истину, если область изображения не является обсолютно белой.

    :param cx: x координата центра области (в пикселях)
    :type cx: int
    :param cy: y координата центра области (в пикселях)
    :type cy: int
    :param cnt: массивв контуров (len == 4x2)
    :type cnt: numpy.ndarray
    :param img: изображение для отрисовки результата
    :param img: :cpp:type:`cv::Mat`
    :return: совпали ли области
    :rtype: bool
    """
    is_white = True
    if img is None:
        if debug_prints:
            print('No grey image! Search by only visual image.')
        return is_white

    check_coords = list()
    check_coords.append([cx, cy])
    dif = 8
    while dif <= 100:
        # Top left
        x = cx - (abs(cnt[0][0] - cx)) / dif
        y = cy - (abs(cnt[0][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        # Bottom left
        x = cx - (abs(cnt[1][0] - cx)) / dif
        y = cy + (abs(cnt[1][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        # Bottom right
        x = cx + (abs(cnt[2][0] - cx)) / dif
        y = cy + (abs(cnt[2][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        # Top right
        x = cx + (abs(cnt[3][0] - cx)) / dif
        y = cy - (abs(cnt[3][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        if abs(x - cx) < 5 or abs(y - cy) < 5:
            break
        dif *= 1.2

    image_ = deepcopy(img)
    for coord in check_coords:
        if 230 <= image_[coord[1]][coord[0]] <= 255:
            is_white = False
    return is_white


def compare_with_depth_area(cx, cy, cnt, img):
    """
    Функция сравнения области изображения с картой глубины.

    :param cx: x координата центра области (в пикселях)
    :type cx: int
    :param cy: y координата центра области (в пикселях)
    :type cy: int
    :param cnt: массивв контуров (len == 4x2)
    :type cnt: numpy.ndarray
    :param img: изображение для отрисовки результата
    :param img: :cpp:type:`cv::Mat`
    :return: совпали ли области
    :rtype: bool
    """
    global depth_image

    is_center = True
    if depth_image is None:
        if debug_prints:
            print ('No depth image! Search by only visual image.')
        return is_center

    check_coords = list()
    check_coords.append([cx, cy])
    dif = 8
    while dif <= 100:
        # Top left
        x = cx - (abs(cnt[0][0] - cx)) / dif
        y = cy - (abs(cnt[0][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        # Bottom left
        x = cx - (abs(cnt[1][0] - cx)) / dif
        y = cy + (abs(cnt[1][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        # Bottom right
        x = cx + (abs(cnt[2][0] - cx)) / dif
        y = cy + (abs(cnt[2][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        # Top right
        x = cx + (abs(cnt[3][0] - cx)) / dif
        y = cy - (abs(cnt[3][1] - cy)) / dif
        check_coords.append([int(x), int(y)])
        if abs(x - cx) < 5 or abs(y - cy) < 5:
            break
        dif *= 1.2

    depth_img = deepcopy(depth_image)
    # depth_img = cv2.blur(depth_img, (100, 100))
    # depth_img = cv2.bilateralFilter(depth_img, 10, 400, 400)
    _retval, bin_depth = cv2.threshold(depth_img, 50, 255, cv2.THRESH_BINARY)
    for coord in check_coords:
        if bin_depth[coord[1]][coord[0]] != 255:
            is_center = False
            cv2.circle(img, (coord[0], coord[1]), 2, (0, 255, 0), -1)
        else:
            cv2.circle(img, (coord[0], coord[1]), 2, (255, 255, 255), -1)

    return is_center


def angle_cos(p0, p1, p2):
    """
    Функция вычисляет косинус угла, заданного 3-мя точками.

    :param p0: точка 1 ([x, y])
    :param p0: numpy.array
    :param p1: точка 2 ([x, y])
    :param p1: numpy.array
    :param p2: точка 3 ([x, y])
    :param p2: numpy.array
    :return: косунус угла
    :rtype: float
    """
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / math.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def debug_found_grey_value(img):
    """
    Функция сохраняет изображения для оценки порогового значения перегонки в бинарное изображение.
    :param img:
    :type img: :cpp:type:`cv::Mat`
    :return:
    """
    # Для поиска пограничного значения перегонки в бинарное изображение
    for i in xrange(0, 255, 1):
        # print ('write %d, image size = %s' % (i, len(img)))
        _retval, bin_img = cv2.threshold(img, 175, i, cv2.THRESH_BINARY)
        cv2.imwrite('img/%d.png' % i, bin_img)
    #     print ('type = %s', len(bin_img[0]))
    #     # cv2.imshow('image', bin_img)
    #     # time.sleep(1.0)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    exit(0)


def find_squares(img):
    """
    Функция поиска прямоугольников на изображении.

    :param img: одноканальное изображение
    :type img: :cpp:type:`cv::Mat`
    :return: массив контуров, являющимися прямоугольниками
    :rtype: list
    """
    global binary_image

    # debug_found_grey_value(img)

    squares = []
    bin_img = deepcopy(binary_image)
    for i in xrange(150, 180, 1):
        _retval, bin_img = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)
        if is_cv2():
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif is_cv3():
            bin_img, contours, _hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = []
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

            image_area = image_width_px * image_height_px  # Площадь всего изображения
            range_false = 2500  # Диапазон ложных прямоугольников
            check_angles = len(cnt) == 4  # Должно быть 4 угла
            check_range = range_false/3 < cv2.contourArea(cnt) < (image_area - range_false)
            if check_angles and check_range and cv2.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                if debug_prints:
                    print ('max_cos = %s' % max_cos)
                if max_cos < 0.3:  # 0  - идеальный прямоугольник
                    squares.append(cnt)

    if bin_img is not None:
        binary_image = deepcopy(bin_img)

    return squares


def fill_window_direction_msg(cx, cy):
    """
    Функция заполнения сообщения смещения центра изображния от центра окна.

    :param cx: x координата центра окна в пикселях
    :type cx: int
    :param cy: y координата центра окна в пикселях
    :type cy: int
    """
    global window_center_direction

    window_center_direction.width_angle = px2rad(cx - image_width_px / 2)
    window_center_direction.height_angle = -px2rad(cy - image_height_px / 2)
    if debug_prints:
        print ('center diff = %s %s' % (degrees(window_center_direction.width_angle),
                                        degrees(window_center_direction.height_angle)))


def process_image(frame):
    """
    Window visual detection.

    :param frame: image in cv2 format
    :type frame: :cpp:type:`cv::Mat`
    :return: image with counters
    :rtype: :cpp:type:`cv::Mat`
    """
    global window_center_direction

    if frame is None:
        return frame

    # Преобразуем в черно-белое изображение
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Размываем изображение
    # grey_img = cv2.blur(grey_img, (10, 5))

    # Поииск прямоугольников
    squares = find_squares(grey_img)
    cv2.drawContours(grey_img, squares, -1, (0, 255, 0), 3)

    # Рисуем пересекающиеся линии в центре изображения
    # cv2.line(grey_img, (image_width_px/2, 0), (image_width_px/2, image_height_px), (0, 255, 0), 2)
    # cv2.line(grey_img, (0, image_height_px/2), (image_width_px, image_height_px/2), (0, 255, 0), 2)

    window_centers = []
    # Ищем центр каждого контура
    for i in range(len(squares)):
        cnt = squares[i]
        # Отфильтровываем прямоугольник, имеющий размер кадра
        image_area = image_width_px * image_height_px
        if (image_area - 1000) < cv2.contourArea(cnt) < image_area:
            continue
        M = cv2.moments(cnt)
        try:
            # Ищем центр контура
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Отбрасываем повторяющиеся центы контуров
            unique = True
            for j in range(len(window_centers)):
                if abs(window_centers[j][0] - cx) < 5 and abs(window_centers[j][1] - cy) < 5:
                    unique = False
            if unique and \
               compare_with_depth_area(cx, cy, cnt, grey_img) and \
               compare_with_grey_img(cx, cy, cnt, grey_img):
                c_xy = [cx, cy]
                window_centers.append(c_xy)
        except ZeroDivisionError:
          pass

    window_center_direction.found_window = False
    count = 0
    res_center = [0, 0]
    for center in window_centers:
        res_center[0] += center[0]
        res_center[1] += center[1]
        count += 1
    try:
        res_center[0] = res_center[0] / count
        res_center[1] = res_center[1] / count
        window_center_direction.found_window = True
        cv2.circle(grey_img, (res_center[0], res_center[1]), 10, (0, 255, 0), -1)
        cv2.line(grey_img, (res_center[0], res_center[1]),
                           (image_width_px/2, image_height_px/2), (0, 255, 0), 3)
        fill_window_direction_msg(res_center[0], res_center[1])
    except ZeroDivisionError:
        window_center_direction.found_window = False

    # Рисуем данные лидаров на изображении
    # grey_img = draw_lidar(grey_img, lidar_vu8)
    # grey_img = draw_lidar(grey_img, lidar_360deg)

    return grey_img


def check_lidar_range(value):
    """
    Функция проверки длины лидара.

    :param value: значение дальности
    :type value: double
    :return:
    :rtype: bool
    """
    if np.isinf(value):
        return True
    return False


def draw_lidar(image, lidar):
    """
    Функция отрисовки лидара на изображении камеры.

    :param image: image for drawing
    :type: image: :cpp:type:`cv::Mat`
    :param lidar: drawing lidar
    :type lidar: Lidar
    :return: Изображение с отрисованным лидаром
    :rtype: :cpp:type:`cv::Mat`
    """
    if lidar.data is None:
        return image

    lidar.sectors_count = len(lidar.data.ranges)

    # Если развертка лидара больше ширины изображения, обрезаем лидар по изображению
    start_ang = -image_width_rad/2 if image_width_rad/2.0 < lidar.width/2.0 else -lidar.width/2.0
    end_ang = image_width_rad/2 if image_width_rad/2.0 < lidar.width/2.0 else lidar.width / 2.0
    sectors_count = int(lidar.sectors_count * (end_ang-start_ang) / lidar.width)
    start_lidar = start_ang - lidar.data.angle_min
    start_sector = start_lidar * lidar.sectors_count / lidar.width

    for i in range(sectors_count):
        x_top = abs(int(rad2px(image_width_rad / 2.0 + start_ang + lidar.data.angle_increment * (sectors_count-i))))
        y_top = abs(int(image_height_px / 2.0 - rad2px(lidar.height / 2.0)))
        x_bottom = abs(int(x_top + rad2px(lidar.data.angle_increment)))
        y_bottom = abs(int(image_height_px / 2.0 + rad2px(lidar.height / 2.0)))
        cv2.rectangle(image, (x_bottom, y_bottom), (x_top, y_top), (0, 255, 0), 5)
        # Берем сектор лидара, соответствующий сектору на изображении
        sec_num = int(start_sector + i)
        if sec_num < 0:
            sec_num += lidar.sectors_count
        # Если значение больше попрогового, отмечаем сектор
        if check_lidar_range(lidar.data.ranges[sec_num]):
            cv2.rectangle(image, (x_bottom, y_bottom), (x_top, y_top), (255, 255, 255), 5)
        # else:
            # cv2.rectangle(image, (x_bottom, y_bottom), (x_top, y_top), (0, 255, 0), 5)

    return image


def deg2px(deg):
    """
    Return pixels from degrees value.

    :param deg: degrees
    :return: pixels
    """
    global image_width_rad
    return deg * image_width_px / degrees(image_width_rad)


def rad2px(rad):
    """
    Return pixels from radians angle value.

    :param rad: radians
    :return: pixels
    """
    return rad * image_width_px / image_width_rad


def px2deg(px):
    """
    Return degrees from pixels value.

    :param px: pixels
    :return: degrees
    """
    global image_width_rad
    return px * degrees(image_width_rad) / image_width_px


def px2rad(px):
    """
    Return radians angle from pixels value.

    :param px: pixels
    :return: radians
    """
    return px * image_width_rad / image_width_px


def main():
    """
    Main function.

    :return: exec code
    :rtype: int
    """
    global cv_image_
    global res_image_
    global binary_image

    # Create ROS node
    rospy.init_node('window_detector_node', anonymous=True)

    # Init subscribe and publisher
    rospy.Subscriber('/stereo/left/image_raw', Image, image_callback)
    rospy.Subscriber('/stereo/disparity', DisparityImage, stereo_depth_callback)
    # rospy.Subscriber('/stereo/points2', PointCloud2, point_cloud_callback)
    # rospy.Subscriber('/lidar_vu8_front/sectored', LaserScan, lidar_vu8_callback)
    rospy.Subscriber('/rplidar/scan', LaserScan, lidar_360deg_callback)

    res_image_pub = rospy.Publisher("window_detector/result_image", Image, queue_size=1)
    depth_image_pub = rospy.Publisher("window_detector/depth_image", Image, queue_size=1)
    bin_image_pub = rospy.Publisher("window_detector/binary_image", Image, queue_size=10)
    angle_dir_pub = rospy.Publisher("window_detector/angle_dir", WindowAngleDir, queue_size=1)

    # srv = Server(WindowDetectorConfig, cfg_callback)

    rate = rospy.Rate(20)
    bridge = CvBridge()
    while not rospy.is_shutdown():
        res_image_ = process_image(deepcopy(cv_image_))

        if depth_image is not None:
            depth_image_pub.publish(bridge.cv2_to_imgmsg(deepcopy(depth_image), encoding="passthrough"))
        if binary_image is not None:
            bin_image_pub.publish(bridge.cv2_to_imgmsg(deepcopy(binary_image), encoding="passthrough"))
        if res_image_ is not None:
            res_image_pub.publish(bridge.cv2_to_imgmsg(res_image_, encoding="passthrough"))
        angle_dir_pub.publish(window_center_direction)
        rate.sleep()
    return 0


if __name__ == '__main__':
    main()
