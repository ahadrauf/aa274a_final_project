#!/usr/bin/env python

import rospy
import os

import pickle
from std_msgs.msg import String, UInt32, Time, Float64, Int8, Header
from nav_msgs.msg import OccupancyGrid, MapMetaData


def callback(data, n):
    print(n[0])
    if (n[0] % 10) == 0: # the map topic refreshes about once every 1.2 seconds
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'map.txt')

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        # with open(filename, 'rb') as infile:
        #     saved_data = pickle.load(infile)
        #     print(saved_data.info)

        rospy.loginfo("Saved map to " + str(filename))
    n[0] += 1

    
def listener():
    rospy.init_node('save_map', anonymous=True)

    n = [0]  # A list so its values are considered mutable
    rospy.Subscriber("/map", OccupancyGrid, lambda data: callback(data, n))

    rospy.spin()

if __name__ == '__main__':
    listener()