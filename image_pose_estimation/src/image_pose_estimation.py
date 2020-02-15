#!/usr/bin/env python
# coding=utf8

import os.path
import cv2
import math
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf
from drone_msgs.msg import WindowPointDir
from visualization_msgs.msg import Marker

import image_processing

bridge = CvBridge()
cv_image = Image()
marker = Marker()

topic_tf_child = "/object"
topic_tf_perent = "/base_link"

windowFoundMsg = WindowPointDir()

t = TransformStamped()
tf2_br = tf2_ros.TransformBroadcaster()
tfBuffer = tf2_ros.Buffer()

OFFSET_COURSE = math.radians(30)
image_proc = None

def image_clb(data):
    """
    get image from ros
    :type data: Image
    :return:
    """
    global bridge, cv_image, get_image_flag, topic_tf_perent

    topic_tf_perent = data.header.frame_id

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        get_image_flag = True
    except CvBridgeError as e:
        print(e)

def pubTf(position, orientation):
    """
    publish find object to tf2
    :param position:
    :param orientation:
    :return:
    """
    global t, topic_tf_perent, topic_tf_child

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = topic_tf_perent
    t.child_frame_id = topic_tf_child
    t.transform.translation.x = position[0]
    t.transform.translation.y = position[1]
    t.transform.translation.z = position[2]
    quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])

    t.transform.rotation.x = quaternion[0]
    t.transform.rotation.y = quaternion[1]
    t.transform.rotation.z = quaternion[2]
    t.transform.rotation.w = quaternion[3]
    print("tf pub")
    tf2_br.sendTransform(t)

def getPointFromMap():
    """
    Возвращаем позицию окна относительно Map
    :return:
    """
    global topic_tf_child, tfBuffer

    try:
        trans = tfBuffer.lookup_transform("map", topic_tf_child, rospy.Time())
        rotXYZ = tf.transformations.euler_from_quaternion((trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w))
        # print("window on map:", trans)
        # print("roll", math.degrees(rotXYZ[0]))
        # print("pitch", math.degrees(rotXYZ[1]))
        # print("yaw", math.degrees(rotXYZ[2]))
        return trans.transform.translation, rotXYZ
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("tf error")
        return None, None

def camera_info_clb(data):
    """
    Get cam info
    :param data:
    :return:
    """
    global image_proc, camera_info_sub
    if image_proc is None:
        return
    t = 0
    for i in range(3):
        for k in range(3):
            image_proc.camera_parameters[i][k] = data.K[i+k]
            t+=1
    for i in range(5):
        image_proc.camera_distortion_param[i] = data.D[i]
    camera_info_sub.unregister()
    print("get camera params")

def pub_windows():
    """
    Публикация найденного окна относительлно map
    :return:
    """
    global windowsPub, windowFoundMsg, markerPub, image_height
    trans, rot = getPointFromMap()
    if trans and rot:
        windowFoundMsg.point.point = trans
        # курс равен pitch
        windowFoundMsg.point.course = rot[2] + (math.pi*0.5)

        windowsPub.publish(windowFoundMsg)

        markerPub.publish(setup_market(trans,windowFoundMsg.point.course, wight=size_image, height=image_height))


def setup_market(pose, yaw, height, wight):
    """
    Настройка маркера для отображения в rviz
    :type pose: DronePose
    :param pose:
    :return:
    """
    global marker
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.get_rostime()
    marker.ns = "windows"
    marker.id = 10
    marker.action = 0

    quaternion = tf.transformations.quaternion_from_euler(0.0,0.0, yaw)
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    marker.scale.x = 0.05
    marker.scale.y = wight
    marker.scale.z = height
    marker.type = Marker.CUBE
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.8
    marker.pose.position.x = pose.x
    marker.pose.position.y = pose.y
    marker.pose.position.z = pose.z
    return marker

def rotate_point3d(rot,xyz_point):
    """
    Поворачиваем на заданный угол относительно a на угол rot
    :param rot: угол порота
    :return: возвращаем точку повёрнутую на нужный угол

    """
    rotate = np.array([[np.cos(rot), -np.sin(rot)],
                       [np.sin(rot), np.cos(rot)]])
    pos = np.array([[xyz_point[0][0]], [xyz_point[2][0]]])
    val = np.dot(rotate,pos)
    return (val[0][0], xyz_point[1][0], val[1][0])

############
### Main ###
############
if __name__ == '__main__':
    rospy.init_node('image_pose_estimation_node', anonymous=True)

    listener = tf2_ros.TransformListener(tfBuffer)

    _rate = 10.                 # this is a voracious application, so I recommend to lower the frequency, if it is not critical

    MIN_MATCH_COUNT = 10        # the lower the value, the more sensitive the filter
    blur_threshold = 300        # the higher the value, the more sensitive the filter
    max_dist = 10.              # publish objects that are no further than the specified value

    size_image = 2.             # the width of the image in meters

    use_image = False           # uses a known image
    image_path = "image.jpg"    # path to known image

    show_image = True           # show image in window
    camera_name = "camera"      # the name of the camera in ROS
    max_angle = 45                # max angle in degree of yaw for find object

    # init params
    camera_name = rospy.get_param("~camera_name", camera_name)
    topic_tf_child = rospy.get_param("~frame_id", topic_tf_child)
    show_image = rospy.get_param("~show_image", show_image)

    use_image = rospy.get_param("~use_image", use_image)
    image_path = rospy.get_param("~image_path", image_path)
    size_image = rospy.get_param("~size_image", size_image)
    max_angle = rospy.get_param("~max_angle", max_angle)

    MIN_MATCH_COUNT = rospy.get_param("~min_match_count", MIN_MATCH_COUNT)
    blur_threshold = rospy.get_param("~blur_threshold", blur_threshold)
    _rate = rospy.get_param("~rate", _rate)
    _rate = -1. if _rate <= 0. else _rate

    image_height = 0.0

    #  Check init params
    if use_image is False and show_image is False:
        rospy.logerr("image not set.\n"
                     "Solutions:\n"
                     "* Enable param: show_image = True\n"
                     "* Set path to image in param: image_path and use_image = true")
        exit()

    if use_image is True:
        if image_path == "" or os.path.isfile(image_path) is False:
            rospy.logerr("Path to image invalid.\n"
                         "Solutions:\n"
                         "* Set path to image in param: image_path\n"
                         "* Disable param: use_image = False")
            exit()

    # init params
    get_image_flag = False

    image_proc = image_processing.ImageEstimation(MIN_MATCH_COUNT, blur_threshold, use_image, size_image, image_path, show_image)
    image_proc.max_dist = max_dist

    rate = rospy.Rate(_rate)
    rospy.Subscriber(camera_name+"/image_raw", Image, image_clb)
    camera_info_sub = rospy.Subscriber(camera_name+"/camera_info", CameraInfo, camera_info_clb)

    # топик публикации найденного окна
    windowsPub = rospy.Publisher("window_detector/point", WindowPointDir, queue_size=10)
    markerPub = rospy.Publisher("window_detector/marker", Marker, queue_size=10)

    while not rospy.is_shutdown():
        if get_image_flag:
            # read the current frame
            k = cv2.waitKey(1) & 0xFF
            frame, trans, rot = image_proc.update(cv_image, k)

            if trans is not None and rot is not None:
                # if trans[2] > size_image: # and math.degrees(abs(rot[2])) < max_angle:
                    image_height = image_proc.heght_image
                    # newTrans = rotate_point3d(OFFSET_COURSE,trans)
                    pubTf(trans, rot)
                    pub_windows()

            if get_image_flag and show_image:
                cv2.imshow("frame", frame)

            # break when user pressing ESC
            if k == 27:
                break
        if _rate > 0.:
            rate.sleep()
    cv2.destroyAllWindows()
