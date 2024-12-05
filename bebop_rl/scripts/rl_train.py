#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import String
from mav_msgs.msg import Actuators

class RLTrain:
    def __init__(self):
        rospy.init_node('rl_train', anonymous=True)
        
        self.rate = rospy.Rate(0.2)
        
        self.motor_cmd_pub = rospy.Publisher('/bebop/command/motors', Actuators, queue_size=10)
        self.reset_world_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_world_service.wait_for_service()

    def run(self):
        while not rospy.is_shutdown():
            motor_cmd = Actuators()
            motor_cmd.angular_velocities = [400, 400, 400, 400]
            self.motor_cmd_pub.publish(motor_cmd)
            rospy.loginfo("Publishing motor command: {}".format(motor_cmd.angular_velocities))
            
            self.reset_world_service()
            rospy.loginfo("Resetting world")
            
            self.rate.sleep()


if __name__ == '__main__':
    try:
        rltrain = RLTrain()
        rltrain.run()
    except rospy.ROSInterruptException:
        pass