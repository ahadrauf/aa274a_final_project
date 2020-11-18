#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
from geometry_msgs.msg import Pose2D
from asl_turtlebot.msg import VendorList, DetectedObject, DetectedObjectList

def callback(data, pub, vendors):
    for name, obj in zip(data.objects, ob_msgs):
        if name in vendors:
            msg, n = vendors[name]
            theta_avg = (obj.thetaleft + obj.thetaright) / 2.0
            x = data.robot_pose.x + obj.distance*np.cos(data.robot_pose.theta + theta_avg)
            y = data.robot_pose.y + obj.distance*np.sin(data.robot_pose.theta + theta_avg)
            msg.x = (n*msg.x + x) / (n + 1)
            msg.y = (n*msg.y + y) / (n + 1)
            msg.theta = (n*msg.theta + theta_avg) / (n + 1)
            vendors[name] = (msg, n+1)
        else:
            theta_avg = (obj.thetaleft + obj.thetaright) / 2.0
            x = data.robot_pose.x + obj.distance*np.cos(data.robot_pose.theta + theta_avg)
            y = data.robot_pose.y + obj.distance*np.sin(data.robot_pose.theta + theta_avg)
            vendors[name] = (Pose2D(x, y, theta_avg), 1)
    pub.publish([pose for pose in ])

    
def listener():
    rospy.init_node('vendor_catalog', anonymous=True)

    vendors = {}

    pub = rospy.Publisher('/vendor_catalog', VendorList, queue_size=10)
    rospy.Subscriber("/detector/objects", DetectedObjectList, lambda data: callback(data, pub, vendors))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()