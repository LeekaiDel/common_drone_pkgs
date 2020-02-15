#!/usr/bin/env python
# coding=utf8

# Глабальный планировщик для работы с одной камерой
import rospy
import numpy as np
import tf
from copy import deepcopy

from geometry_msgs.msg import Pose
from geometry_tools import *
from move_base_msgs.msg import MoveBaseActionFeedback
from actionlib_msgs.msg import GoalStatus, GoalID

from drone_msgs.msg import *
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse

from dynamic_reconfigure.server import Server
from drone_global_planner.cfg import globalCfgConfig
from debug_tools import *
from modes import *

"""
Настраиваемые параметры
"""
show_debug = True                             # плаг отображения отладочной информации (траектории, маркеров)

use_rotate_flag = False                       # Если True, то дрон летит зигзагообразно
rate_rotate = 0.2                            # Скорость поворота (в hz)
max_angle_rotate = 15.0                      # Максимальный угол поворота  (+- в градусах)

tolerance_windows_course = math.radians(60)   # допустимый угол(в градусах) по курсу для обнаружения окна
tolerance_windows_height = 2.0                # допустимая высота до окна от траектории
tolerance_to_goal = 0.7                       # допустимое расстояние для переключения другой точки
tolerance_to_path = 0.7                       # допустимое расстояние до траектории для переключения целевой точки
move_offset = 3.0                             # дистанция смещения
use_global_planner = True                        # флаг для использования move_base для получения траектории
path_index_offset = 25                               # следующая точка на путитраектории


active_flag = True              # Статус работы планировщика
status_goal_flag = False        # Если false, то, указываем конечную точку через rviz

# Список топиков
goal_topic = "/goal_pose"
pose_topic = "/mavros/local_position/pose"
activeStatusTopic = "/drone/global_planner/active"
activeSrvTopic = "/drone/global_planner/set_active"
setGoalStatusTopic = "/drone/global_planner/goal_status"
setGoalServiceTopic = "/drone/global_planner/set_goal"
windowPoint_topic = "/window_detector/point"


"""
Start params
"""
# distances
dist_normal_to_path = 0.0       # растояние прекции дрона до троектории
dist_to_target = 0.0            # среднеквадротичное расстояние до цели
dist_normal_to_target_on_path = 0.0    # растояние на траектории

dist_normal_to_end = 0.0        # среднеквадротичное расстояние до цели
# dist_on_path = 0.0              # растояние на траектории
dist_target_to_end = 0.0        # растояние от целевой точки на прямой до конечной
dist_window_to_path = 0.0       # растояние от окна до траектории


# goal params
start_point_3d = np.array([0.0, 0.0, 2.0])    # начальная точка
end_point_3d = np.array([0.0, 0.0, 2.0])      # конечная точка
target_point_on_path_3d = np.array([0.0, 0.0, 2.0])   # целевая точка на траектории
target_point_3d = np.array([0.0, 0.0, 2.0])   # целевая точка на траектории
drone_on_path_point_3d = np.array([0.0, 0.0, 0.0])  # точка  на траектории
vector_of_path = np.array([0.0, 0.0, 0.0])    # нормализованный вектор по курсу движения
current_pose_3d = np.array([0.0, 0.0, 0.0])   # нормализованный вектор по курсу движения
vector_offset = np.array([0.0, 0.0, 0.0])     # вектор смещения по оси Y
window_pose_3d = np.array([0.0, 0.0, 0.0])    # позиция окна
window_on_path = np.array([0.0, 0.0, 0.0])    # позиция окна на траектории
window_relative_path = np.array([0.0, 0.0, 0.0]) # позиция окна относительно траектории

current_course = 0.0                          # текущий курс
path_course = 0.0                             # курс траектории
error_course = 0.0                            # Ошибка по курсу, текущий курс - курс траектории
SIN_CORRECTION = 1.0 / math.pi          # поправка для синуса (нужна чтобы перевести rad к еденице

# режимы работы
current_mode = Fly_modes.INIT_STATE         # текущий режим работы
mission_state = Mission_state.WAIT          # состояние планировщика

# drone params
current_pose = Pose()                         # текущее положение дрона
course_offset = 0.0                           # смещение курса курса дрона к окну (не 0 если курс дрона не папаллелен окну)
end_point_goal = Goal()                       # конечная точка    course задаёт направление траектории движения
target_point = Goal()                    # тукущая желаемая точка
window_found = WindowPointDir()               # Сообщение найденного окна

# Nodes params
init_start_point_flag = False                 # Флаг инициализации начального положения.
print_delay = 1.0                             # задержка перед публикацией статусной информации

globalPath = Path()
localPath = Path()
globalPathIndex = 0.0

cfg_srv = None
init_server = False
pathStatus = GoalStatus.ACTIVE

"""
Функция считывания параметров планировщика
"""
def cfg_callback(config, level):
    """
    Callback считываения с rqt
    :param config:
    :param level:
    :return:
    """
    global init_server
    global active_flag, status_goal_flag, \
        use_global_planner, use_rotate_flag, rate_rotate, max_angle_rotate, \
        tolerance_windows_course, tolerance_windows_height, tolerance_to_goal, tolerance_to_path, move_offset, path_index_offset
    if init_server is False:
        init_server = True
        return config

    # Параметры регулятора
    active_flag = config["active"]
    status_goal_flag = config["set_goal"]
    use_global_planner = config["use_global_planner"]
    use_rotate_flag = config["use_rotate"]
    rate_rotate = config["rate_rotate"]
    max_angle_rotate = config["max_angle_rotate"]
    tolerance_windows_course = math.radians(config["tolerance_windows_course"])
    tolerance_windows_height = config["tolerance_windows_height"]
    tolerance_to_goal = config["tolerance_to_goal"]
    tolerance_to_path = config["tolerance_to_path"]
    move_offset = config["move_offset"]
    path_index_offset = config["path_index_offset"]


    return config

def set_config(config):
    global active_flag, status_goal_flag, \
        use_global_planner, use_rotate_flag, rate_rotate, max_angle_rotate, \
        tolerance_windows_course, tolerance_windows_height, tolerance_to_goal, tolerance_to_path, move_offset, path_index_offset

    config.update_configuration({'active': active_flag,
                                  'set_goal': status_goal_flag,
                                 'use_global_planner': use_global_planner,
                                'use_rotate' : use_rotate_flag,
                                 'rate_rotate': rate_rotate,
                                 'max_angle_rotate' : max_angle_rotate,
                                 'tolerance_windows_course': tolerance_windows_course,
                                 'tolerance_windows_height':tolerance_windows_height,
                                 'tolerance_to_goal':tolerance_to_goal,
                                 'tolerance_to_path':tolerance_to_path,
                                 'move_offset':move_offset,
                                 'path_index_offset': path_index_offset})


"""
Методы рос
"""

def GoalPoseClb(data):
    """
    Функция получения координат цели
    :type data: Goal
    :param data:
    :return:
    """
    global target_point,\
        status_goal_flag, \
        end_point_goal, end_point_3d, path_course, use_global_planner,active_flag

    if active_flag is False:
        print("Гланировщик выключен!")
        return

    if status_goal_flag is False:
        # if use_global_planner:
        #     pubGlobalPoint(data.pose.point)

        end_point_goal = data
        end_point_3d = point_to_np(end_point_goal.pose.point)
        path_course = end_point_goal.pose.course
        rospy.loginfo("set end point")

def pubGlobalPoint(data):
    """
    Публикация позиции в глобальный планировщик
    : type data: Point
    :return:
    """
    global pathStatus
    if pathStatus == GoalStatus.PENDING:
        print("pathStatus = GoalStatus.PENDING")
        return

    goalMsg = PoseStamped()
    goalMsg.header.frame_id = "map"
    goalMsg.pose.position = deepcopy(data)
    goalMsg.pose.orientation.w = 1.
    move_base_pub.publish(goalMsg)
    print("set path on move_base")
    pathStatus = GoalStatus.PENDING

def activeSrv(req):
    """
    Change state of planner
    """
    global active_flag, cfg_srv

    active_flag = req.data
    print("Change state:", active_flag)
    resp = SetBoolResponse()
    resp.success = active_flag

    if cfg_srv is not None:
        # write in dyn. rec.
        cfg_srv.update_configuration({'active': active_flag})

    return resp

def setPointSrv(req):
    """
        Изменяем поведение маркера
    """
    global status_goal_flag, cfg_srv
    status_goal_flag = req.data
    if status_goal_flag:
        print("Траектория задана !")
    else:
        print("Траектория не задана !Режим задария траектории.")

    resp = SetBoolResponse()
    resp.success = True

    if cfg_srv is not None:
        # write in dyn. rec.
        cfg_srv.update_configuration({'set_goal': status_goal_flag})
    return resp

def CurrentPoseCb(data):
    """
    Получаем положение дрона
    :param data: PaseStamped дрона
    :return:
    """

    global current_pose, current_pose_3d, init_start_point_flag, start_point_3d, current_course, end_point_3d
    current_pose = data.pose
    current_pose_3d = point_to_np(data.pose.position)
    current_course = get_yaw_from_quat(data.pose.orientation)

    # назначаем начальную точку в текущем положении + заданная высота
    if init_start_point_flag is False:
        start_point_3d[0] = current_pose_3d[0]
        start_point_3d[1] = current_pose_3d[1]
        start_point_3d[2] = end_point_3d[2]
        print("start_point_3d",start_point_3d)
        init_start_point_flag = True

def StatusPathClb(data):
    """
    Статус move base
    """
    global pathStatus
    pathStatus = data.status.status

def PubGoalPoint(vector_3d, course=None):
    """
    Публикация точки на прямой
    :param vector_3d: Желаемая точка
    :param course:  Желаемый курс
    :param path_flag: Флаг, для публикации точки из траектории
    :param target_point_3d: текущая траектория
    :return:
    """
    global target_point, goalPub, target_point_3d, target_point_on_path_3d, use_global_planner

    if use_global_planner is False:
        target_point_on_path_3d = vector_3d

    target_point_3d = vector_3d
    target_point.pose.point = np_to_point(vector_3d)

    if course is not None:
        target_point.pose.course = course

    goalPub.publish(target_point)
    # if use_global_planner:
    #     pubGlobalPoint(np_to_point(target_point_3d))


def globalPathCb(data):
    """
    get global path
    :return:
    """
    global globalPath, globalPathIndex, path_index_offset
    print("global: ", len(data.poses))
    if len(data.poses) > path_index_offset:
        globalPathIndex = int(path_index_offset)
        globalPath = data
    else:
        globalPathIndex = len(data.poses) - 1

def windowPointCb(data):
    """
    Callback найденного окна
    :param data:
    :return:
    """
    global window_pose_3d, window_found, status_goal_flag, end_point_3d, \
        window_relative_path, tolerance_windows_course, path_course, current_mode, active_flag

    if status_goal_flag or active_flag is False:
        return

    course = path_course - data.point.course

    if abs(course) > tolerance_windows_course:
        print("windows course error: {:.3}".format(math.degrees(course)))
        return

    print("windows detection !")

    window_found = data
    window_pose_3d = point_to_np(data.point.point)

    # считаем смещение окна от траектории
    solve_window_relative_path()

    # проверяем условие, удовлетворяет ли найденное окно высоте от траектории
    if abs(window_relative_path[2]) > tolerance_windows_height:
        print("window_offset fail !. height: {:.3}".format(window_relative_path[2]))
        return

    # если всё ОК, то переходим в режим задания смещения
    current_mode = Fly_modes.FIND_WINDOW

def goal_offset_clb(data):
    """
    Смещение траектории, включается только если задана траектория
    :type data: Pose

    :return:
    """
    global status_goal_flag, active_flag, path_course, target_point_on_path_3d, end_point_3d, path_course, current_mode

    if active_flag is False:
        print ("Offset error! Планировщик выключен.")
        return

    if status_goal_flag is False:
        print ("Offset error! Траектория не задана.")
        return

    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion((data.orientation.x,
                                                                   data.orientation.y,
                                                                   data.orientation.z,
                                                                   data.orientation.w))
    offset_yaw = (path_course + yaw) % (np.pi*2.)
    offset_pose = np.array([data.position.x, data.position.y, data.position.z])


    # задаём смещение
    end_point_3d += offset_pose
    target_point_on_path_3d += offset_pose
    path_course = offset_yaw

    end_point_goal.pose.point = np_to_point(end_point_3d)

    if use_global_planner:
        pubGlobalPoint(np_to_point(target_point_on_path_3d))
    print("Goal offset completed!")
    current_mode = Fly_modes.INIT_STATE


"""
Методы планировщика
"""

def solve_values():
    """
    Расчёт значений. Дистанции, позиции и т.д.
    :return:status_goal_flag
    """

    global end_point_goal, end_point_3d, vector_of_path, path_course, current_pose_3d, drone_on_path_point_3d
    global dist_normal_to_path, dist_normal_to_target_on_path, dist_to_target, dist_normal_target, dist_normal_to_end, dist_target_to_end
    global current_course, error_course, vector_offset, target_point_3d, target_point_on_path_3d

    end_point_vector = end_point_3d + vector_of_path

    # получаем точку к траектории

    # считаем вектор направления траектории
    vector_of_path = np.array(rotate_vect(path_course, 1.0))
    vector_offset = np.array(rotate_vect(path_course+(np.pi*0.5), 1.0))
    drone_on_path_point_3d = np.array(normalToPath_3d(end_point_3d, end_point_vector, current_pose_3d))

    # расчитываем дистанцию
    dist_to_target = dist(target_point_3d, current_pose_3d)  # среднеквадротичное расстояние до цели
    dist_normal_to_target_on_path = dist(drone_on_path_point_3d, target_point_on_path_3d, flag_2d=True)  # расстояние на траектории
    dist_normal_to_path = dist(drone_on_path_point_3d, current_pose_3d, flag_2d=True) # растояние прекции дрона до троектории
    dist_normal_target = dist(drone_on_path_point_3d, target_point_on_path_3d, flag_2d=True) # растояние до цели на траектории
    dist_normal_to_end = dist(drone_on_path_point_3d, end_point_3d)  # расстояние дрона на прямой до конечной точки
    dist_target_to_end = dist(target_point_on_path_3d, end_point_3d)  # от целевой точки до конечной точки

    # считаем ошибку между курсом
    error_course = path_course - current_course

def solve_window_relative_path():
    """
    Растёч значений для окна. Ищем точку на прямой, а также её положение относительно прямой в доль курса. Где Y - это смещение
    :return:
    """
    global window_pose_3d, window_on_path, vector_of_path, end_point_3d, window_relative_path, path_course

    window_on_path = np.array(normalToPath_3d(end_point_3d, end_point_3d+vector_of_path, window_pose_3d))
    # ищем с какой стороны от траектории лежит окно
    window_relative_path = np.array(rotate_point2d(-path_course, window_pose_3d - window_on_path))
    window_relative_path[2] = end_point_3d[2] - window_pose_3d[2]
    # print("window_local_pose", window_relative_path)

def move_forvard(dist):
    """
    добавляем сммещение на задонное расстояние в доль страектории
    :param dist:
    :return:
    """
    global vector_of_path
    print("move forvard")
    return vector_of_path * dist

def print_status():
    """
    Печать статусной информации
    :return:
    """
    print("Mode: %s" %(current_mode.name))
    print("Distance:")
    print("--to target:\t\t\t{:.3}".format(dist_to_target))
    print("pathStatus :", pathStatus)
    print("--normal to path:\t\t{:.3}".format(dist_normal_to_path))
    print("--normal to target:\t\t{:.3}".format(dist_normal_target))
    print("--normal to end:\t\t{:.3}".format(dist_normal_to_end))
    print("--target to end:\t\t{:.3}".format(dist_target_to_end))

def getOffset(offset):
    """
    задаём смещение конечной точки на задонной расстояние по оки Y
    :param offset:
    :return:
    """

    global vector_offset
    return vector_offset * offset

def course_rotate(time, hz,maxAngle, pathCourse = 0.0):
	"""
	Поварачиваем целевую точку на заданный курс
	:param time:
	:param hz:
	:param maxAngle:
	:param pathCourse:
	:return: угол
	"""
	return pathCourse + math.sin(time*hz/SIN_CORRECTION)* math.radians(maxAngle)

"""
Основной метод
"""
if __name__ == '__main__':
    rospy.init_node("drone_global_planner_node", anonymous=False)
    rate = rospy.Rate(10.0)


    """
    init dynamic reconfigure server
    """
    cfg_srv = Server(globalCfgConfig, cfg_callback)
    set_config(cfg_srv)

    """
    Subscriber
    """
    # подписываемся на текущую целевую точку
    rospy.Subscriber(goal_topic, Goal, GoalPoseClb)
    # подписываемся на текущую позицию дрона
    rospy.Subscriber(pose_topic, PoseStamped, CurrentPoseCb)
    # найденное окно
    rospy.Subscriber(windowPoint_topic, WindowPointDir, windowPointCb)
    rospy.Subscriber("/drone/global_planner/goal_offset", Pose, goal_offset_clb)

    """
    Publisher
    """
    activePub = rospy.Publisher(activeStatusTopic, Bool, queue_size=10)
    statusPub = rospy.Publisher(setGoalStatusTopic, Bool, queue_size=10)
    goalPub = rospy.Publisher(goal_topic, Goal, queue_size=10)

    """
    Service
    """
    # сервис для отключения планировщика
    rospy.Service(activeSrvTopic, SetBool, activeSrv)
    # сервис смены режима планировщика
    rospy.Service(setGoalServiceTopic, SetBool, setPointSrv)

    # подписаться на глобальный планировщик
    if use_global_planner:
        rospy.Subscriber("move_base/NavfnROS/plan", Path, globalPathCb)
        rospy.Subscriber("/move_base/feedback", MoveBaseActionFeedback, StatusPathClb)
        move_base_pub = rospy.Publisher("move_base_simple/goal", PoseStamped, queue_size=10)
        resetPub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
    # инициализация топиков отладки
    if show_debug:
        draw_path = DrawDebug()
        draw_path.Initialise(path_topic = "/drone/global_planner/path", normal_topic = "/drone/global_planner/path_norm")
        window_path = DrawDebug()
        window_path.Initialise(normal_topic="/drone/global_planner/path_window")
        markers = MarkerDebug()

    # init timer
    old_time = rospy.get_time()
    _timer = 0.0
    _time_offset = 0.0
    _timer_run = 0.0
    """
    Loop
    """
    while not rospy.is_shutdown():

        # публикуем статус планировщика
        activePub.publish(active_flag)
        statusPub.publish(status_goal_flag)

        if not active_flag:
            rate.sleep()
            continue

        dt = rospy.get_time() - old_time
        old_time = rospy.get_time()
        _timer += dt
        _timer_run += dt

        solve_values()  # расчитываем значения для планировщика

        # вращаем курс
        if use_rotate_flag:
            newCourse = course_rotate(time=_timer_run-_time_offset, hz=rate_rotate, pathCourse=path_course,
                                      maxAngle=max_angle_rotate)
        else:
            newCourse = None

        # если изменяет конечню точку, то переключаем режим
        if status_goal_flag is False:
            current_mode = Fly_modes.INIT_STATE

        if current_mode == Fly_modes.INIT_STATE:
            if status_goal_flag is True:
                init_start_point_flag = False
                current_mode = Fly_modes.TAKE_OFF
                _time_offset = _timer_run

        elif current_mode == Fly_modes.TAKE_OFF:
            """
            Летим с начальную точку
            """
            # current_mode = Fly_modes.GO_TO_LINE

            if init_start_point_flag:
                PubGoalPoint(start_point_3d, course=newCourse)

                # Если мы взлетели, то летим к близжайшей точки на траектории
                # Если достигли точки с заданной погрешностью, то меняем режим
                if dist_to_target < tolerance_to_goal:
                    current_mode = Fly_modes.GO_TO_LINE
                    target_point_on_path_3d = drone_on_path_point_3d
                    PubGoalPoint(target_point_on_path_3d, course=newCourse)
                    if use_global_planner:
                        pubGlobalPoint(np_to_point(target_point_on_path_3d))

        elif current_mode == Fly_modes.GO_TO_LINE:
            """
            Летим по траектории2.0
            """

            # публикуем точку пути
            if use_global_planner:
                if len(globalPath.poses) == 0 or globalPathIndex < path_index_offset:
                    newPoint = target_point_on_path_3d
                    print("path == 0")
                    if pathStatus != GoalStatus.PENDING:
                        pathStatus = GoalStatus.ABORTED
                        pubGlobalPoint(np_to_point(target_point_on_path_3d + move_forvard(1.0)))
                        print("update global path")
                else:
                    newPoint = point_to_np(globalPath.poses[globalPathIndex].pose.position)
                    newPoint[2] = end_point_3d[2]
                    print("print globalPath != 0")

                PubGoalPoint(newPoint, course=newCourse)
            # публикуем точку вдоль траектории
            else:
                PubGoalPoint(target_point_on_path_3d, course=newCourse)

            if dist_normal_to_target_on_path < tolerance_to_goal:
                print("set move offset")
                target_point_on_path_3d += move_forvard(move_offset)

                if use_global_planner and pathStatus != GoalStatus.PENDING:
                    resetPub.publish(GoalID())
                    globalPath.poses.clear()
                    pubGlobalPoint(np_to_point(target_point_on_path_3d))

        elif current_mode == Fly_modes.FIND_WINDOW:
            """
            Найдено окно, добавляем смещение
            """

            end_point_3d += getOffset(window_relative_path[1])
            target_point_on_path_3d += getOffset(window_relative_path[1])

            end_point_goal.pose.point = np_to_point(end_point_3d)

            if use_global_planner:
                pubGlobalPoint(np_to_point(target_point_on_path_3d))
            print("windows offset completed!")
            current_mode = Fly_modes.GO_TO_LINE

        # выводим статус планировщика
        if _timer > print_delay:
            print_status()
            _timer = 0.0

        # рисуем траекторию и нормаль к ней а также маркеры
        if show_debug:
            # if current_mode != current_mode.GO_TO_LINE:
            draw_path.DrawPathToGoal(goal=end_point_goal, dist=dist_normal_to_end)
            # else:
            #     draw_path.DrawPathToGoal(point=np_to_point(target_point_on_path_3d),dist=dist_normal_to_end, course=path_course)

            draw_path.DrawPath2(np_to_point(drone_on_path_point_3d), current_pose.position)
            window_path.DrawPath2(np_to_point(window_pose_3d), np_to_point(window_on_path))
            markers.drawMarkers(_droneOnPathPose=drone_on_path_point_3d, _goalOnPathPose=target_point_on_path_3d, _tolerance=tolerance_to_goal)
        rate.sleep()
