#!/usr/bin/env python
import rospy
import os

import pickle
from std_msgs.msg import String, UInt32, Time, Float64, Int8, Header
from nav_msgs.msg import OccupancyGrid, MapMetaData

def talker():
    pub = rospy.Publisher('/map_nav', OccupancyGrid, queue_size=10)
    rospy.init_node('broadcast_map', anonymous=True)
    rate = rospy.Rate(1) # 1 hz
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'map.txt')

    while not rospy.is_shutdown():
        with open(filename, 'rb') as infile:
            saved_map = pickle.load(infile)
            pub.publish(saved_map)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass