#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs import Image

from cv_bridge import CvBridge

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


def image_sub_class:
    def __init__(self):
        
        self.image=[]
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("tb3_0/camera/rgb/image_raw", Image, callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def callback(data):
        self.image=data
        
if __name__ == '__main__':
    image_sub=image_sub_class()
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_sub.image, desired_encoding='passthrough')

