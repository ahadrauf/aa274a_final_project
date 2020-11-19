# AA 274A Final Project, Autumn 2020
#### Purple Tier Pavonebot (Ahad Rauf, Xiangbing Ji, Josiah Wong, Maggie Ford)

### Launch files
 - ```roslaunch asl_turtlebot project_sim.launch``` - for baseline, includes manual waypoint navigation (I think - TBD @Josiah)
 - ```roslaunch asl_turtlebot autonomous_navigation.launch``` - for RRT exploration, saves the map every 10 samples (~every 10 seconds) to a file (map.txt by default)
 - ```roslaunch asl_turtlebot delivery.launch``` - for automated delivery management, broadcasts the map from map.txt every second to a new topic /map_nav

### Extra Scripts Written for Baseline Management
 - ```scripts/add_robot_pose_to_image.py``` - Because detector_mobilenet.py takes a while to finish parsing the latest camera image for vendor signs, a fast-moving car could see a significant error in its XY-position if its pose was sampled after image recognition. This file adds the robot's ```geometry_msgs/Pose2D``` pose to the ```camera/raw_image``` topic that gets sent to detector_mobilenet.py as a cohesive unit, improving positional accuracy by up to 0.5m in testing.
 - ```scripts/aggregate_statistics.py``` - This is a helper function that allows one to aggregate the (mean, std, max, min, num_samples) on any ROS topic with type ```std_msgs/Float32```. It's useful for coming up with metrics in the final presentation.
 - ```scripts/save_map.py``` and ```scripts/broadcast_map.py``` - These are functions that allow for modularity between the active SLAM and delivery phases. In theory, delivery could be baked into the active SLAM segment with little difficulty, but this approach made distributing the workload a little easier.
 - ```scripts/vendor_manager.py``` - The central vendor manager script. It reads the ```/detector/objects``` topic from detector_mobilenet.py and parses the objects' positions to form an internal listing of all the vendors. It also serves as the highest level of the delivery FSM, implementing high-level decision planning on which vendors should be traveled to first.

### Extensions
 - Self navigation - Based on RRT Exploration (https://github.com/hasauino/rrt_exploration), and defined largely by the ```src/rrt_exploration``` and ```src/asl_turtlebot/params``` and ```src/asl_turtlebot/launch/include``` folders
 - High-level decision planning - Based in ```scripts/vendor_manager.py```
 - Used SMACH for FSM creation
 - Ability to navigate duplicate vendors
 - TBD: Obstacle avoidance/AR Tag pose estimation