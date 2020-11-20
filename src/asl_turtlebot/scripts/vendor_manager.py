#!/usr/bin/env python

"""
Top-Level State FSM that oversees high-level task planning.

There are 4 main states:

1) EXPLORE: During this phase, the robot is actively exploring the environment (this may be manual exploration or
    automated). During this phase, new food locations should be detected / logged.

2) PICKUP: During this phase, the robot is travelling to a specified vendor to pickup (and wait for) an order.

3) RETURN: During this phase, the robot returns to its home base location.

4) IDLE: During this phase, the robot remains idle until a new command is received.

Note that this should be the single entry point for all relevant sensory / command data. This node interacts with
    other nodes in the following ways:

    SUBSCRIBES TO:
    /cmd_nav: Receive 2D Pose Navigation commands from rviz
    /delivery_request: Receive array of vendor names from request_publisher
    /detector: Receive detected vendor information from CNN detector

    PUBLISHES TO:
    /cmd_nav_exec: 2D Pose Navigation commands to be executed by navigator.py

    RELEVANT PARAMS:
    /nav/status (String): status of the current navigation command being executed, one of {'completed', 'active', 'failed'}
    /nav/home (2-list): [x, y] home location for turtlebot
"""

import roslib
import rospy
import smach
from smach import Sequence
import smach_ros

import numpy as np

from collections import deque
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
from asl_turtlebot.msg import DetectedObjectList, DetectedObject

# TASK PARAMS
TOTAL_VENDORS = 5
AVERAGING_SIZE = 3

# TOPICS
WAYPOINT_CMD_TOPIC = '/cmd_nav'
DELIVERY_REQUEST_TOPIC = '/delivery_request'
VENDOR_DETECTED_TOPIC = '/detector/objects'
CMD_NAV_EXEC_TOPIC = '/cmd_nav_exec'

# PARAMS
NAV_STATUS_PARAM = '/nav/status'
HOME_POS_PARAM = '/nav/home'

# EXTENSIONS
# Flag to turn on/off the extension to consider vendors with duplicate names. E.g two apple vendors.
DUPLICATE_VENDOR_NAMES_EXIST = True
# For any two detections, how close they are for us to consider they are detections of the same vendor.
SAME_VENDOR_THRESHOLD = 0.1
# The number of detections we need to confirm a vendor actually exists. Otherwise we consider these detections as
# outliers and discard the detected vendor.
NON_OUTLIER_SIZE = 20

# Define all state outcomes
STATE_OUTCOMES = {
    "EXPLORE":  ['completed'],
    "PICKUP":   ['completed'],
    "ORDER":    ['completed'],
    "RETURN":   ['completed'],
    "IDLE":     ['completed'],
    "SM":       ['completed'],
}


class RingBuffer(object):
    """
    Simple RingBuffer object to hold values to average (useful for, e.g.: filtering D component in PID control)

    Note that the buffer object is a 2D numpy array, where each row corresponds to
    individual entries into the buffer

    Args:
        dim (int): Size of entries being added. This is, e.g.: the size of a state vector that is to be stored
        length (int): Size of the ring buffer
    """

    def __init__(self, dim, length):
        # Store input args
        self.dim = dim
        self.length = length
        self.size = 0 # how many elements in this buffer now.

        # Save pointer to current place in the buffer
        self.ptr = 0

        # Construct ring buffer
        self.buf = np.zeros((length, dim))

    def push(self, value):
        """
        Pushes a new value into the buffer

        Args:
            value (int or float or array): Value(s) to push into the array (taken as a single new element)
        """
        # Add value, then increment pointer
        self.buf[self.ptr] = np.array(value)
        self.ptr = (self.ptr + 1) % self.length
        self.size = self.size + 1 if self.size < self.length else self.length

    def clear(self):
        """
        Clears buffer and reset pointer
        """
        self.buf = np.zeros((self.length, self.dim))
        self.ptr = 0

    @property
    def average(self):
        """
        Gets the average of components in buffer

        Returns:
            float or np.array: Averaged value of all elements in buffer
        """
        return np.sum(self.buf, axis=0) / self.size

# Define vendor manager class
class VendorManager:
    def __init__(self):
        # Initialize this as a node
        rospy.init_node("vendor_manager")
        rospy.loginfo("Initializing...")

        # Initialize relevant attributes
        self.vendor_pos_buffer = {}         # Used to calculate the moving average of a vendor pos
        self.delivery_list = None           # Will be added when receive vendor list

        # Subscribers
        rospy.Subscriber(WAYPOINT_CMD_TOPIC, Pose2D, self.cmd_nav_callback)
        rospy.Subscriber(DELIVERY_REQUEST_TOPIC, String, self.delivery_request_callback)
        rospy.Subscriber(VENDOR_DETECTED_TOPIC, DetectedObjectList, self.vendor_detected_callback)

        # Publishers
        self.cmd_exec_pub = rospy.Publisher(CMD_NAV_EXEC_TOPIC, Pose2D, queue_size=10)

        # Create the SMACH state machine
        self.sm = Sequence(outcomes=STATE_OUTCOMES['SM'], connector_outcome='completed')

        # Add appropriate states
        with self.sm:
            Sequence.add('EXPLORE', self.Explore(self))
            Sequence.add('PICKUP', self.Pickup(self))
            Sequence.add('RETURN', self.Return(self))
            Sequence.add('IDLE', self.Idle(self))

    def run(self):
        """
        Main run loop
        """
        rospy.loginfo("Running vendor manager SM!")
        self.sm.execute()

    def cmd_nav_callback(self, data):
        """
        Determines what to do with newly received navigation command based on current state

        Args:
            data (Pose2D): Received message data
        """
        # We only pass this received command to the navigator if we're in the explore state
        if self.sm.get_active_states()[0] == 'EXPLORE':
            goal = [data.x, data.y, data.theta]
            self.navigate_to_goal(goal, wait_until_completed=False)

    def vendor_detected_callback(self, data):
        """
        Processes the newly received vendor information and stores it internally

        Args:
            data (DetectedObjectList): Received message data
        """
        # Get robot x, y, theta
        x, y, th = data.robot_pose.x, data.robot_pose.y, data.robot_pose.theta
        #rospy.loginfo("Robot pos: ({}, {}, {})".format(x, y, th))
        # Loop through all detected objects (each of type DetectedObjectList)
        if not DUPLICATE_VENDOR_NAMES_EXIST:
            for obj_msg in data.ob_msgs:
                # Use the detected vendor information and the current state of the robot to calculate the position (x,y)
                # of the vendor in the world frame.
                # d_theta = wrap_to_0_to_2pi(thetaright + wrap_to_0_to_2pi(thetaright - thetaleft) / 2)
                d_theta = (obj_msg.thetaright + ((obj_msg.thetaleft - obj_msg.thetaright) % (2 * np.pi)) * 0.5) % (2 * np.pi)
                vendor_x = x + obj_msg.distance * np.cos(th + d_theta)
                vendor_y = y + obj_msg.distance * np.sin(th + d_theta)
                #rospy.loginfo("Vendor d: {}, th_avg: {}".format(obj_msg.distance, d_theta))

                # Add RingBuffer if new vendor is being detected
                if obj_msg.name not in self.vendor_pos_buffer:
                    self.vendor_pos_buffer[obj_msg.name] = RingBuffer(dim=2, length=30)

                self.vendor_pos_buffer[obj_msg.name].push(np.array((vendor_x, vendor_y)))

        if DUPLICATE_VENDOR_NAMES_EXIST:
            for obj_msg in data.ob_msgs:
                # Use the detected vendor information and the current state of the robot to calculate the position (x,y)
                # of the vendor in the world frame.
                # d_theta = wrap_to_0_to_2pi(thetaright + wrap_to_0_to_2pi(thetaright - thetaleft) / 2)
                d_theta = (obj_msg.thetaright + ((obj_msg.thetaleft - obj_msg.thetaright) % (2 * np.pi)) * 0.5) % (
                            2 * np.pi)
                vendor_x = x + obj_msg.distance * np.cos(th + d_theta)
                vendor_y = y + obj_msg.distance * np.sin(th + d_theta)
                detected_vendor_pos = np.array((vendor_x, vendor_y))
                # rospy.loginfo("Vendor d: {}, th_avg: {}".format(obj_msg.distance, d_theta))

                # Add RingBuffer if new vendor is being detected
                # print('keys', self.vendor_pos_buffer.keys())
                if obj_msg.name not in self.vendor_pos_buffer:
                    # print('name not in buffer', obj_msg.name, detected_vendor_pos)
                    self.vendor_pos_buffer[obj_msg.name] = [RingBuffer(dim=2, length=30)]
                    self.vendor_pos_buffer[obj_msg.name][0].push(detected_vendor_pos)

                # If there is already an vendor with the same name detected. Then there are two possibility:
                # 1) the new detection belongs to one of the existing vendors, if the new detection is within the
                # threshold radius of one of the existing vendor's position.
                # 2) the new detection doesn't belong to any of the existing vendors.
                #
                # Note self.vendor_pos[obj_msg.name] is a list of RingBuffer now. Each entry in this list represents a
                # RingBuffer of a unique vendor although all vendors in this list share the same vendor name.
                if obj_msg.name in self.vendor_pos_buffer:
                    existing_vendor = False
                    for idx, existing_vendor_pos in enumerate(self.vendor_pos[obj_msg.name]):
                        print(detected_vendor_pos, existing_vendor_pos, np.linalg.norm(detected_vendor_pos - existing_vendor_pos))
                        if np.linalg.norm(detected_vendor_pos - existing_vendor_pos) < SAME_VENDOR_THRESHOLD:
                            self.vendor_pos_buffer[obj_msg.name][idx].push(detected_vendor_pos)
                            existing_vendor = True
                            break
                    if not existing_vendor:
                        self.vendor_pos_buffer[obj_msg.name].append(RingBuffer(dim=2, length=30))
                        self.vendor_pos_buffer[obj_msg.name][-1].push(detected_vendor_pos)

        # print('Vendor Pos Dict', self.final_vendor_pos)
        # print('Buffer', self.vendor_pos_buffer[obj_msg.name])
        # print('Vendor Pos', self.vendor_pos[obj_msg.name])


        # Debug
         #rospy.loginfo("{} pos: {}".format(obj_msg.name, self.vendor_pos_buffer[obj_msg.name].average))

    def delivery_request_callback(self, data):
        """
        Processes the received delivery request

        Args:
            data (String): Received message data
        """

        self.delivery_list = data.data.split(",")

    def shutdown_callback(self):
        """
        Execute any final commands before shutting down
        """
        pass

    @property
    def vendor_pos(self):
        if not DUPLICATE_VENDOR_NAMES_EXIST:
            return {name: buf.average for name, buf in self.vendor_pos_buffer.items()}

        if DUPLICATE_VENDOR_NAMES_EXIST:
            vendor_pos = {}
            for name, buf_list in self.vendor_pos_buffer.items():
                vendor_pos[name] = [buf.average for buf in buf_list]
            return vendor_pos

    @property
    def final_vendor_pos(self):
        if not DUPLICATE_VENDOR_NAMES_EXIST:
            return self.vendor_pos

        if DUPLICATE_VENDOR_NAMES_EXIST:
            final_vendor_pos = {}
            for name, buf_list in self.vendor_pos_buffer.items():
                final_buf_list = []
                for buf in buf_list:
                    # If a vendor receives less than "NON_OUTLIER_SIZE" number of detections, consider this vendor as an
                    # outlier and remove it from the final_vendor_pos.
                    if buf.size >= NON_OUTLIER_SIZE:
                        final_buf_list.append(buf)
                final_vendor_pos[name] = [buf.average for buf in final_buf_list]
            return final_vendor_pos

    class Explore(smach.State):
        """
        Exploration state. Returns when we've seen all vendors and we receive a vendors list for deliveries
        """
        def __init__(self, outer):
            self.outer = outer
            smach.State.__init__(self, outcomes=STATE_OUTCOMES['EXPLORE'])

        def execute(self, userdata):
            rospy.loginfo("Executing state EXPLORE")
            while self.outer.delivery_list is None:# or len(self.outer.vendor_pos.keys()) != TOTAL_VENDORS:
                rospy.sleep(1)

            return "completed"

    class Pickup(smach.State):
        """
        Pickup state. Plans and executes path to go to each requested vendor
        """
        def __init__(self, outer):
            self.outer = outer
            smach.State.__init__(self, outcomes=STATE_OUTCOMES['PICKUP'])

        def execute(self, userdata):
            rospy.loginfo("Executing state PICKUP")
            # Plan path
            vendors = self.outer.plan_delivery(self.outer.delivery_list)
            # Execute pickup at each vendor location sequentially
            for vendor in vendors:
                if vendor not in self.outer.final_vendor_pos:
                    print("sorry, the requested vendor is not found!")
                    return "failed"

                # Travel to this location
                if not DUPLICATE_VENDOR_NAMES_EXIST:
                    self.outer.navigate_to_goal(
                        [self.outer.vendor_pos[vendor][0], self.outer.vendor_pos[vendor][1], 0])    # TODO: What to use for th value?

                if DUPLICATE_VENDOR_NAMES_EXIST:
                    if not self.outer.final_vendor_pos[vendor]:
                        print("sorry, the requested vendor is not found!")
                        return "failed"
                    # Always navigate to the first vendor if there are duplicates. For example, if the user requests
                    # pick up apple and we have 3 apple vendors "Apple1", "Apple2" and "Apple3". We always go to
                    # "Apple1" to pick up the apple. This is incorrect, ideally we should implement a shortest path
                    # planner to decide which apple vendor we will navigate to pick up.
                    self.outer.navigate_to_goal(
                        [self.outer.final_vendor_pos[vendor][0][0], self.outer.vendor_pos[vendor][0][1], 0])

                # Now, "receive" the order for 3 seconds
                rospy.sleep(3)

            return "completed"

    class Return(smach.State):
        """
        Return state. Returns to home location
        """
        def __init__(self, outer):
            self.outer = outer
            smach.State.__init__(self, outcomes=STATE_OUTCOMES['RETURN'])

        def execute(self, userdata):
            rospy.loginfo("Executing state RETURN")
            home_pos = rospy.get_param(HOME_POS_PARAM)
            home_goal = [home_pos[0], home_pos[1], 0]       # TODO: What to use for theta?
            # Travel to home
            self.outer.navigate_to_goal(home_goal)

            return "completed"

    class Idle(smach.State):
        """
        Idle state. Does nothing, this should be when we have completed our delivery task!
        """
        def __init__(self, outer):
            self.outer = outer
            smach.State.__init__(self, outcomes=STATE_OUTCOMES['IDLE'])

        def execute(self, userdata):
            rospy.loginfo("Executing state IDLE")
            rospy.loginfo("Completed delivery task! :D")
            # Loop endlessly
            while True:
                continue

            return "completed"

    def navigate_to_goal(self, goal, wait_until_completed=True):
        """
        Navigates to the requested @goal. Returns when the goal has (presumably) succeeded.

        Args:
            goal (3-array): (x, y, th) global values to reach
            wait_until_completed (bool): If true, does not return until goal has been reached.
        """
        rospy.set_param(NAV_STATUS_PARAM, 'active')
        cmd = Pose2D()
        cmd.x, cmd.y, cmd.theta = goal
        # Travel to this location
        self.cmd_exec_pub.publish(cmd)
        # Wait until this path is completed if requested
        if wait_until_completed:
            while rospy.get_param(NAV_STATUS_PARAM) != 'completed':
                rospy.sleep(1)

    def plan_delivery(self, delivery_list):
        """
        Calculates the best order to reach all delivery locations

        Args:
            delivery_list (list of str): List of vendors to reach

        Returns:
            list: optimized ordered vendor list to travel to
        TODO: Make this shortest path delivery
        """
        return delivery_list


if __name__ == '__main__':
    vm = VendorManager()
    rospy.on_shutdown(vm.shutdown_callback)
    vm.run()
