#!/usr/bin/env python
# coding=utf8

from math import sqrt
from collections import deque

import rospy
from drone_msgs.msg import Goal
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped


file_path = 'waypoints.txt'  # path to file with waypoints
drone_pose = None  # current drone position
cur_wp = None  # current waypoint
tolerance_dist = 0.1  # meters


def read_waypoints(file_path):
    """
    Return waypoints list which read from file.

    :type file_path:
    :rtype: deque
    """
    waypoints = deque()

    wp_file = open(file_path, "r")
    lines = wp_file.readlines()
    for line in lines:
        coords = line.split(" ")
        if len(coords) < 3:
            continue
        point = Point()
        point.x = float(coords[0])
        point.y = float(coords[1])
        point.z = float(coords[2])
        waypoints.append(point)

    return waypoints


def get_dist(p1, p2):
    """
    Return distance between two points

    :type p1: geometry_msgs.msg.Point
    :type p2: geometry_msgs.msg.Point
    :rtype: float
    """
    return sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2)


def completed(waypoint):
    """
    Check drone if drone to fly to waypoint or waypoint is first

    :type waypoint: geometry_msgs.msg.Point
    :rtype: bool
    """
    global drone_pose
    global tolerance_dist

    if drone_pose is None:
        return False

    if waypoint is None:  # waypoint is first
        return True

    dist2wp = get_dist(drone_pose.pose.position, waypoint)
    return dist2wp <= tolerance_dist


def pose_cb(pose):
    """
    Drone pose callback

    :type pose: Point
    """
    global drone_pose
    drone_pose = pose


def goal_cb(goal):
    """
    Drone goal callback

    :type goal: Goal
    """
    global cur_wp
    cur_wp = goal.pose.point


def goal_publish(publisher, point):
    """
    Publish goal to drone

    :param publisher: rospy.Publisher
    :param point: Point
    """
    print ('PUB')
    goal = Goal()
    goal.ctr_type = Goal.POSE
    goal.pose.point = point

    publisher.publish(goal)


def main():
    global cur_wp
    global file_path

    rospy.init_node("waypoint_switcher_node")

    pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb)
    goal_sub = rospy.Subscriber("/goal_pose", Goal, goal_cb)
    goal_pub = rospy.Publisher("/goal_pose", Goal, queue_size=1)

    waypoints = read_waypoints(file_path)

    print (waypoints)

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        try:
            if completed(cur_wp):
                cur_wp = waypoints.popleft()
                goal_publish(goal_pub, cur_wp)
        except IndexError:
            continue

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        exit(1)
