#!/usr/bin/env python

import rospy
import os

from std_msgs.msg import Float32
from asl_turtlebot.msg import AggregateStatistics

def callback(data, name, pub_list, value_list):
    current_stats = value_list[name]
    data = data.data
    old_avg = current_stats.avg
    current_stats.avg = (current_stats.n*current_stats.avg + data) / (current_stats.n + 1)

    # Welford's Online Algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    current_stats.std = ((current_stats.n*current_stats.std) + (data - old_avg)*(data - current_stats.avg))/(current_stats.n + 1)


    current_stats.max = max(current_stats.max, data)
    current_stats.min = min(current_stats.min, data)
    current_stats.n += 1

    value_list[name] = current_stats
    pub_list[name].publish(current_stats)

    
def listener():
    rospy.init_node('aggregate_statistics', anonymous=True)

    topics = ['/robot/distance_after_detection']
    pub_list = {topic: rospy.Publisher(topic + "_pub", AggregateStatistics, queue_size=10) for topic in topics}
    value_list = {topic: AggregateStatistics(topic, 0, 0., 0., -1e3, 1e3) for topic in topics}

    for key in pub_list:
    	rospy.Subscriber(key, Float32, lambda data: callback(data, key, pub_list, value_list))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()