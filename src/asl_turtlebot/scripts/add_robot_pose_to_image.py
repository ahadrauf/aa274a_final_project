#!/usr/bin/env python
# license removed for brevity
import rospy
from asl_turtlebot.msg import ImagePose
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
import tf2_ros
import tf_conversions

def callback(data, pub, tfBuffer):
    trans = None
    while trans is None:
        try:
            trans = tfBuffer.lookup_transform('map', 'base_camera', rospy.Time(), rospy.Duration(1.0))  # TransformStamped
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("ERROR: tfBuffer couldn't detect transformation between /map and /base_camera in add_robot_pose_to_image.py")
            continue

    q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
    e = tf_conversions.transformations.euler_from_quaternion(q)  # roll, pitch, yaw = theta
    pose = Pose2D(trans.transform.translation.x, trans.transform.translation.y, e[-1])
    ret = ImagePose(data, pose)
    pub.publish(ret)
    
def listener():
    rospy.init_node('image_with_pose_modifier', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    pub = rospy.Publisher('/camera/image_raw_with_pose', ImagePose, queue_size=10)
    rospy.Subscriber("/camera/image_raw", Image, lambda data: callback(data, pub, tfBuffer))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()