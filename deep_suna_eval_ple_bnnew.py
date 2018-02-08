# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames
# Software License Agreement (BSD License)
#
# Copyright (c) 2017, SHIBAURA INSTUTUE OF TECHNOLOGY
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2017,SHIBAURA INSTUTUE OF TECHNOLOGY
# Revision $Id$

"""
Deep reinforcement learning.
@author: Sun Zeyuan, Nakatani Masayuki, Uchimura yu
"""
import simple
import ple
import tensorflow as tf
import numpy as np
import cv2


def _variable_on_gpu(name, shape, initializer):

    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_gpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _batch_normalization(x):
    """Batch normalization."""
    with tf.variable_scope("batch_normal"):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))

        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
        # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())

    return y, beta, gamma


class DeepSuna:
    ACTIONS_COUNT = 4  # 3 number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 50000  # time steps to observe before training
    EXPLORE_STEPS = 1000000  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.10  # final chance of an action being random
    MINI_BATCH_SIZE = 50  # size of mini batches
    STATE_FRAMES = 4  # number of frames to store in the state
    RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (40, 40)
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
    LEARN_RATE = 1e-5
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
    DECAY_STEPS = 200000
    START_TIME = 0

    def __init__(self, checkpoint_path="deep_suna_networks"):
        """
        Example of deep q network for pong

        :param checkpoint_path: directory to store checkpoints in
        :type checkpoint_path: str
        """

        self._time = self.START_TIME
        self._checkpoint_path = checkpoint_path

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = None

        # Create an optimizer that performs gradient descent.
        self.opt = tf.train.AdamOptimizer(self.LEARN_RATE)

        self._input_states = tf.placeholder("float",
                                            [None, self.RESIZED_SCREEN_X,
                                             self.RESIZED_SCREEN_Y,
                                             self.STATE_FRAMES])
        # with tf.device('/gpu:0'):
        with tf.variable_scope('conv1'):
            self.kernel1 = _variable_with_weight_decay('weights',
                                                       shape=[8, 8, self.STATE_FRAMES, 32],
                                                       stddev=0.01,
                                                       wd=None)
            self.biases1 = _variable_on_gpu('biases', [32], tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(self._input_states, self.kernel1, [1, 2, 2, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, self.biases1)
            batch_norm, self.beta1, self.gamma1 = _batch_normalization(pre_activation)
            conv1 = tf.nn.relu(batch_norm)

        with tf.variable_scope('conv2'):
            self.kernel2 = _variable_with_weight_decay('weights',
                                                       shape=[4, 4, 32, 64],
                                                       stddev=0.01,
                                                       wd=0.0)
            self.biases2 = _variable_on_gpu('biases', [64], tf.constant_initializer(0.01))
            # conv2
            conv = tf.nn.conv2d(conv1, self.kernel2, [1, 2, 2, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, self.biases2)
            batch_norm, self.beta2, self.gamma2 = _batch_normalization(pre_activation)
            conv2 = tf.nn.relu(batch_norm)

        with tf.variable_scope('conv3'):
            self.kernel3 = _variable_with_weight_decay('weights',
                                                       shape=[3, 3, 64, 64],
                                                       stddev=0.01,
                                                       wd=0.0)
            self.biases3 = _variable_on_gpu('biases', [64], tf.constant_initializer(0.01))
            # conv3
            conv = tf.nn.conv2d(conv2, self.kernel3, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, self.biases3)
            batch_norm, self.beta3, self.gamma3 = _batch_normalization(pre_activation)
            conv3 = tf.nn.relu(batch_norm)

        with tf.variable_scope('local3'):
            self.weights4 = _variable_with_weight_decay('weights', shape=[6400, 256],
                                                        stddev=0.01, wd=0.0)
            self.biases4 = _variable_on_gpu('biases', [256], tf.constant_initializer(0.01))
            # local3
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(conv3, [-1, 6400])
            # dim = reshape.get_shape()[1].value
            fully_connected = tf.matmul(reshape, self.weights4) + self.biases4
            # batch_norm = _batch_normalization(fully_connected, 256, 2)
            local3 = tf.nn.relu(fully_connected)
            # _activation_summary(local3)
            # fiction_dropout = tf.nn.dropout(local3, keep_prob=0.5)

        with tf.variable_scope('softmax_linear'):
            self.weights6 = _variable_with_weight_decay('weights', [256, self.ACTIONS_COUNT],
                                                        stddev=0.01, wd=0.0)
            self.biases6 = _variable_on_gpu('biases', [self.ACTIONS_COUNT],
                                            tf.constant_initializer(0.01))
            self.output_layer = tf.add(tf.matmul(local3, self.weights6), self.biases6)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        self.replay_number = 0
        self.terminal = True

        self.restore_filename = 100000
        self.full_file_eval = True
        self.file_folder = "networks_v10_ple_bnnew_180125"
        self.initial_env_number = 6

        self.environment = simple.Agent()
        self.env = ple.PLE(self.environment, fps=10, force_fps=self.full_file_eval, display_screen=True)
        self.ple_action_list = self.env.getActionSet()

        self.saver = tf.train.Saver()
        self.saver.restore(self._session, self.file_folder + "/model" + str(self.restore_filename))
        if self.full_file_eval:
            self.stepsfile = open(self.file_folder + "/Stepsdata.txt", "w")

    def game_learning(self):

        screen_array = self.env.getScreenRGB()
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array,
                                                            (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y)),
                                                 cv2.COLOR_BGR2GRAY)
        # screen_resized_grayscaled *= 1/255

        if self.terminal:
            keynum = self.environment.keynum
            print ("Steps per episode = ", keynum)
            if self.full_file_eval:
                self.stepsfile.write(str(keynum) + "\n")

            self.replay_number += 1
            self.env.reset_game()
            self._last_state = np.stack(tuple(screen_resized_grayscaled for _ in range(self.STATE_FRAMES)), axis=2)
        else:
            screen_resized_binary = np.reshape(screen_resized_grayscaled,
                                               (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y, 1))
            current_state = np.append(self._last_state[:, :, 1:], screen_resized_binary, axis=2)
            # update the old values
            self._last_state = current_state
            # print self._last_state.shape

        if self.full_file_eval and self.replay_number > self.initial_env_number:
            self.restore_filename += 100000
            self.saver.restore(self._session, self.file_folder + "/model"
                               + str(self.restore_filename))
            print ("restore file name", self.restore_filename)
            self.stepsfile.write("Start" + str(self.restore_filename) + "\n")
            self.replay_number = 1

        self.terminal = self.env.game_over()

        # show resized image

        # cv2.imshow("show", screen_resized_grayscaled)
        # cv2.waitKey(0)

        self._last_action = np.zeros([self.ACTIONS_COUNT])
        new_action = self._choose_next_action(self._last_state)
        self._last_action[new_action] = 1

        self.env.act(self.ple_action_list[new_action])

    def _choose_next_action(self, state):
        self.readout_t = \
            self._session.run(self.output_layer, feed_dict={self._input_states: [state]})[0]

        action_index = np.argmax(self.readout_t)
        return action_index


if __name__ == '__main__':
    player = DeepSuna()
    # player.start()
    while True:
        player.game_learning()
