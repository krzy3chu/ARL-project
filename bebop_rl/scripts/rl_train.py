#!/usr/bin/env python

import random

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import String
from mav_msgs.msg import Actuators
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates


# Constants
MAX_POSITION = 5
MIN_POSITION = -5


class RLTrain:
    def __init__(self):
        rospy.init_node("rl_train", anonymous=True)

        self.rate = rospy.Rate(0.25)

        # bebop state
        self.bebop_state = ModelState()

        # gazebo bebop state subscriber
        self.bebop_state_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.bebop_state_callback
        )

        # motor command publisher
        self.motor_cmd_pub = rospy.Publisher(
            "/bebop/command/motors", Actuators, queue_size=10
        )

        # reset world service
        self.reset_world_service = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.reset_world_service.wait_for_service()

        # set model state service
        self.set_model_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        self.set_model_state_service.wait_for_service()

    def bebop_state_callback(self, data):
        try:
            idx = data.name.index("bebop")
            self.bebop_state.model_name = "bebop"
            self.bebop_state.pose = data.pose[idx]
            self.bebop_state.twist = data.twist[idx]
            self.bebop_state.reference_frame = "world"
        except ValueError:
            rospy.logwarn("Bebop model not found in ModelStates")

    def spawn_bebop(self):
        # reset world
        self.reset_world_service()

        # spawn bebop at random position
        new_state = ModelState()
        new_state.model_name = "bebop"
        new_state.pose.position.x = random.uniform(MIN_POSITION, MAX_POSITION)
        new_state.pose.position.y = random.uniform(MIN_POSITION, MAX_POSITION)
        new_state.pose.position.z = random.uniform(0, MAX_POSITION)
        self.set_model_state_service(new_state)
        
        rospy.loginfo("Resetting world, spawning Bebop at random position")

    def train_loop(self):
        while not rospy.is_shutdown():
            # spawn bebop at random position
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
