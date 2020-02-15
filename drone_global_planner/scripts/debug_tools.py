#!/usr/bin/env python
# coding=utf8

# Класс для отрисоки отладочной информации планироващика

import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from geometry_tools import *
from drone_msgs.msg import Goal
from visualization_msgs.msg import MarkerArray, Marker

class DrawDebug():
    def __init__(self, _frame_id = "map"):

        # 1 path params
        self.path_pose_1 = PoseStamped()
        self.path_pose_1.pose.orientation.w = 1
        self.path_pose_2 = PoseStamped()
        self.path_pose_2.pose.orientation.w = 1
        self.pathMsg = Path()
        self.pathPub = None

        # 1.2
        self.dist_ray = 20.0

        self.frame_id =_frame_id
        # 2 path params
        self.pose_on_path = PoseStamped()
        self.pose_on_path.pose.orientation.w = 1
        self.pose_on_space = PoseStamped()
        self.pose_on_space.pose.orientation.w = 1
        self.pathNormalMsg = Path()
        self.pathNormalPub = None

    def Initialise(self, path_topic = None, normal_topic = None):


        if path_topic:
            self.pathMsg.header.frame_id = self.frame_id
            self.pathPub = rospy.Publisher(path_topic, Path, queue_size=10)

        if normal_topic:
            self.pathNormalMsg.header.frame_id = self.frame_id
            self.pathNormalPub = rospy.Publisher(normal_topic, Path, queue_size=10)



    def DrawPath(self,start_point, finish_point):
        """
        Отрисовка начального и конечного пути

        :param start_point: Начальная точка
        :type start_point: Point
        :param finish_point: Конечная точка
        :type finish_point: Point
        """
        self.path_pose_1.pose.position = start_point
        self.path_pose_2.pose.position = finish_point

        self.pathMsg.poses = [self.path_pose_1, self.path_pose_2]
        self.pathPub.publish(self.pathMsg)
        print("send path")

    def DrawPath2(self, on_path_point = None, on_space_point = None):
        """
        Отрисовка нормализованного вектора к траектории
        :param on_path_point: Начальная точка
        :type on_path_point: Point
        :param on_space_point: Конечная точка
        :type on_space_point: Point
        """
        if on_space_point is None or on_space_point is None:
            self.pathNormalMsg.poses = []
            self.pathNormalPub.publish(self.pathNormalMsg)
            return
        
        self.pose_on_path.pose.position = on_path_point
        self.pose_on_space.pose.position = on_space_point

        self.pathNormalMsg.poses = [self.pose_on_path, self.pose_on_space]
        self.pathNormalPub.publish(self.pathNormalMsg)

    def DrawNormalToPath(self,point):
        """
        Отрисовка перпендикуляра к траектории
        :param point: Точка в пространстве
        :type point: Point
        """
        self.pose_on_space.pose.position = point

        # метод расчёта нормали к траеткори
        pointNorm = normalToPath_goal(self.path_pose_1.pose.position, self.path_pose_2.pose.position, point)

        self.pose_on_path.pose.position = pointNorm
        self.pathNormalMsg.poses = [self.pose_on_path, self.pose_on_space]
        self.pathNormalPub.publish(self.pathNormalMsg)

    def DrawPathToGoal(self, goal = None, point = None, dist = None, course=None):
        """
        Отрисовка начального и конечного пути

        :param start_point: Начальная точка
        :type start_point: Point
        :param finish_point: Конечная точка
        :type finish_point: Point
        """
        # self.path_pose_1.pose.position = goal.pose.point
        if dist == None:
            dist = 5
        if course != None and point != None:
            offset = rotate_vect(course, dist + 5)
        else:
            offset = rotate_vect(goal.pose.course, dist+5)
            point = goal.pose.point

        self.path_pose_1.pose.position.x = point.x + offset[0]
        self.path_pose_1.pose.position.y = point.y + offset[1]
        self.path_pose_1.pose.position.z = point.z

        self.path_pose_2.pose.position.x = point.x - offset[0]
        self.path_pose_2.pose.position.y = point.y - offset[1]
        self.path_pose_2.pose.position.z = point.z

        self.pathMsg.poses = [self.path_pose_1, self.path_pose_2]
        self.pathPub.publish(self.pathMsg)


class MarkerDebug():
    """
    Класс публикации маркеров
    """
    def __init__(self, _frame_id="map", _topic="/drone/global_planner/markers"):
        self.frame_id = _frame_id
        self.topic = _topic

        self.markersMsg = MarkerArray()
        self.droneOnPath = Marker()
        self.goalOnPath = Marker()
        self.goalTolerance = Marker()
        self.tolerance = 0.0

        self.markerPub = rospy.Publisher(self.topic, MarkerArray, queue_size=10)

    def update(self):
        self.markersMsg = [self.droneOnPath, self.goalOnPath, self.goalTolerance]

    def setup_marker(self, pose, name, rgba, size = 0.2, flatFlag = False):
        """
        Настройка маркера для отображения в rviz
        :type pase: Point
        :param pose: Vector
        :return:
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.get_rostime()
        marker.ns = name
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1.0

        marker.scale.x = size
        marker.scale.y = size
        if flatFlag:
            marker.scale.z = 0.01
        else:
            marker.scale.z = size

        marker.type = Marker.SPHERE
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = pose[2]

        return marker

    def drawMarkers(self, _droneOnPathPose = None, _goalOnPathPose = None, _tolerance = None):

        if _tolerance is not None:
            self.goalTolerance = _tolerance

        if _droneOnPathPose is not None:
            self.droneOnPath = self.setup_marker(_droneOnPathPose, "drone_on_path", [1.0,0.,0.,1.])

        if _goalOnPathPose is not None:
            self.goalOnPath = self.setup_marker(_goalOnPathPose, "goat_on_path", [1.,1.,0,1.])
            self.goalTolerance = self.setup_marker(_goalOnPathPose, "goal_tolerance", [1.0,1.,0.,0.2], size=self.goalTolerance, flatFlag=True)

        self.update()
        self.markerPub.publish(self.markersMsg)