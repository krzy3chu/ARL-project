#!/usr/bin/env python

import random

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import String
from mav_msgs.msg import Actuators
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


# Constants
MAX_POSITION = 5
MIN_POSITION = -5


class RLTrain:
    def __init__(self):
        rospy.init_node("rl_train", anonymous=True)

        self.rate = rospy.Rate(0.2)

        self.motor_cmd_pub = rospy.Publisher(
            "/bebop/command/motors", Actuators, queue_size=10
        )
        self.reset_world_service = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.reset_world_service.wait_for_service()

        self.set_model_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        self.set_model_state_service.wait_for_service()

        self.i = 0

    def spawn_bebop(self):
        # reset world
        self.reset_world_service()
        rospy.loginfo("Resetting world")

        # spawn Bebop at random position
        new_state = ModelState()
        new_state.model_name = "bebop"
        new_state.pose.position.x = random.uniform(MIN_POSITION, MAX_POSITION)
        new_state.pose.position.y = random.uniform(MIN_POSITION, MAX_POSITION)
        new_state.pose.position.z = random.uniform(0, MAX_POSITION)
        self.set_model_state_service(new_state)
        rospy.loginfo("Spawning Bebop at random position")

    def train_loop(self):
        while not rospy.is_shutdown():
            # spawn Bebop at random position
            self.spawn_bebop()

            # set motor command
            motor_cmd = Actuators()
            motor_cmd.angular_velocities = [400, 400, 400, 400]
            self.motor_cmd_pub.publish(motor_cmd)
            rospy.loginfo(
                "Publishing motor command: {}".format(motor_cmd.angular_velocities)
            )

            # wait defined time
            self.rate.sleep()


if __name__ == "__main__":
    try:
        rltrain = RLTrain()
        rltrain.train_loop()
    except rospy.ROSInterruptException:
        pass
