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

TODO (Xiangbing?):
    A) Convert DetectedObject info into global (x,y) coordinates for a given vendor
"""

import roslib
import rospy
import smach
from smach import Sequence
import smach_ros

from collections import deque
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
from asl_turtlebot.msg import DetectedObject

# TASK PARAMS
TOTAL_VENDORS = 4           # TODO: This needs to be updated once we have a final list
AVERAGING_SIZE = 3

# TOPICS
WAYPOINT_CMD_TOPIC = '/cmd_nav'
DELIVERY_REQUEST_TOPIC = '/delivery_request'
VENDOR_DETECTED_TOPIC = '/detector'
CMD_NAV_EXEC_TOPIC = '/cmd_nav_exec'

# PARAMS
NAV_STATUS_PARAM = '/nav/status'
HOME_POS_PARAM = '/nav/home'

# Define all state outcomes
STATE_OUTCOMES = {
    "EXPLORE":  ['completed'],
    "PICKUP":   ['completed'],
    "ORDER":    ['completed'],
    "RETURN":   ['completed'],
    "IDLE":     ['completed'],
    "SM":       ['completed'],
}

# Define vendor manager class
class VendorManager:
    def __init__(self):
        # Initialize this as a node
        rospy.init_node("vendor_manager")
        rospy.loginfo("Initializing...")

        # current state of the robot
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Initialize relevant attributes
        self.vendor_pos = {}                # Will be added to as we encounter vendors. Should be (x,y) value per vendor
        self.vendor_pos_buffer = {}         # Used to calculate the moving average of a vendor pos
        self.delivery_list = None           # Will be added when receive vendor list

        # Subscribers
        rospy.Subscriber(WAYPOINT_CMD_TOPIC, Pose2D, self.cmd_nav_callback)
        rospy.Subscriber(DELIVERY_REQUEST_TOPIC, String, self.delivery_request_callback)
        rospy.Subscriber(VENDOR_DETECTED_TOPIC, DetectedObject, self.vendor_detected_callback)

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
        rospy.loginfo("current state: {}".format(self.sm.get_active_states()[0]))
        self.x, self.y, self.theta = data.x, data.y, data.theta

        if self.sm.get_active_states()[0] == 'EXPLORE':
            goal = [data.x, data.y, data.theta]
            self.navigate_to_goal(goal, wait_until_completed=False)

    def vendor_detected_callback(self, data):
        """
        Processes the newly received vendor information and stores it internally

        Args:
            data (DetectedObject): Received message data
        """
        # Use the detected vendor information and the current state of the robot to calculate the position (x,y)
        # of the vendor in the world frame.
        theta_avg = (data.thetaleft + data.thetaright) / 2.0
        vendor_x = self.x + data.distance * np.cos(self.theta + theta_avg)
        vendor_y = self.y + data.distance * np.sin(self.theta + theta_avg)

        # calculate moving average of the vendor position and save the updated position in self.vendor_pos
        if data.id in self.vendor_pos_buffer:
            buffer = self.vendor_pos_buffer[data.id]
            if len(buffer) == AVERAGING_SIZE:
                buffer.popleft()
            buffer.append((vendor_x, vendor_y))
        else:
            self.vendor_pos_buffer[data.id] = deque((vendor_x, vendor_y))

        self.vendor_pos[data.id] = self.vendor_pos_buffer[data.id] / len(self.vendor_pos_buffer[data.id])


    def delivery_request_callback(self, data):
        """
        Processes the received delivery request

        Args:
            data (String): Received message data
        """

        self.delivery_list = data.split(",")

    def shutdown_callback(self):
        """
        Execute any final commands before shutting down
        """
        pass

    class Explore(smach.State):
        """
        Exploration state. Returns when we've seen all vendors and we receive a vendors list for deliveries
        """
        def __init__(self, outer):
            self.outer = outer
            smach.State.__init__(self, outcomes=STATE_OUTCOMES['EXPLORE'])

        def execute(self, userdata):
            rospy.loginfo("Executing state EXPLORE")
            while self.outer.delivery_list is None or len(self.outer.vendor_pos.keys()) != TOTAL_VENDORS:
                rospy.sleep(1)

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
                # Travel to this location
                self.outer.navigate_to_goal([self.outer.vendor_pos[vendor][0], self.outer.vendor_pos[vendor][1], 0])    # TODO: What to use for th value?
                # Now, "receive" the order for 3 seconds
                rospy.sleep(3)

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
