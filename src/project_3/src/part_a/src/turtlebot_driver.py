#!/usr/bin/env python
# license removed for brevity

#Referenced from: http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
'''

def callback(data):
    #rospy.loginfo("I heard %s",data.data)
    return data.clock.secs+data.clock.nsecs*1e-9
    
def listener():
    rospy.init_node('clock_sub')
    rospy.Subscriber("chatter", Clock, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
'''
def talker():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.init_node('my_initials')
    vel_msg=Twist()
    rate = rospy.Rate(10) # 10hz
    init_time=rospy.get_time()
    
    velocity=0.2
    while not rospy.is_shutdown():

        delay=rospy.get_time()-init_time

        #delay=delay+90
        if(delay<=7.8):
            vel_msg.angular.z=3.142/8 #pi/8 rad/s #
            vel_msg.linear.x=0
        elif(delay>4 and delay<=35):
            vel_msg.linear.x=velocity*25/35 #
            vel_msg.angular.z=0
        elif(delay>35 and delay<=39):
            vel_msg.angular.z=-3.142/8 #pi/8 rad/s #
            vel_msg.linear.x=0
        elif(delay<=45):
            vel_msg.linear.x=0.5*velocity #
            vel_msg.angular.z=0
        elif( delay<=51):
            vel_msg.angular.z=3.142/8 #pi/8 rad/s #
            vel_msg.linear.x=0
        elif(delay<54):
            vel_msg.linear.x=0.5*velocity #
            vel_msg.angular.z=0
        elif(delay<=60):
            vel_msg.angular.z=-3.142/8 #3pi/8 rad/s
            vel_msg.linear.x=0
        elif(delay<=63):
            vel_msg.angular.z=0
            vel_msg.linear.x=0.5*velocity
        elif(delay<=69):
            vel_msg.angular.z=3.142/8
            vel_msg.linear.x=0
        elif(delay<=73):
            vel_msg.angular.z=0
            vel_msg.linear.x=velocity
        else:
            vel_msg.linear.x=0
            vel_msg.angular.z=0

            
        hello_str = "hello world %s" % rospy.get_time()
        #rospy.loginfo(hello_str)
        pub.publish(vel_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass