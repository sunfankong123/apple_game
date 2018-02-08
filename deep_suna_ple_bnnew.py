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

from __future__ import print_function

"""
Deep reinforcement learning.
@author: Sun Zeyuan, Nakatani Masayuki, Uchimura yu
"""
import os
import random
from collections import deque
import re
import sys
import signal

import time

import simple
import ple
import tensorflow as tf
import numpy as np
import cv2


def _activation_summary(x):
    """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_gpu(name, shape, initializer):
    """Helper to create a Variable stored on GPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
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


def _target_batch_normalization(x):
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


start_time = time.time()
play_steps = 0
random_time = 0


def handler(signal, frame):
    print ("Keyboard interrupted")
    finish_time = time.time()
    processing_time = finish_time - start_time
    print ("Processing time " + format(processing_time) + "seconds")
    time_per_step = processing_time / play_steps
    print ("Learning time per step" + format(time_per_step) + "seconds")
    full_random_duration = random_time - start_time
    print ("Full random running time" + format(full_random_duration) + "seconds")
    sys.exit(0)


signal.signal(signal.SIGINT, handler)


class DeepSuna:
    ACTIONS_COUNT = 4  # 3 number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 50000  # time steps to observe before training
    EXPLORE_STEPS = 1000000  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.10  # final chance of an action being random
    MEMORY_SIZE = 1000000  # number of observations to remember
    TARGET_NETWORK_UPDATE_FREQ = 10000  # target update frequency
    MINI_BATCH_SIZE = 50  # size of mini batches
    STATE_FRAMES = 4  # number of frames to store in the state
    RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (40, 40)
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
    SAVE_EVERY_X_STEPS = 100000
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

        # pygame.init()
        self.environment = simple.Agent()
        self.env = ple.PLE(self.environment, display_screen=False)
        self.ple_action_list = self.env.getActionSet()

        # self.env.init()

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = None

        global_step = tf.Variable(0, trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        self.learning_rate = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                                        global_step,
                                                        self.DECAY_STEPS,
                                                        self.LEARNING_RATE_DECAY_FACTOR,
                                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        self.opt = tf.train.AdamOptimizer(self.LEARN_RATE)

        # Calculate the gradients for each model tower.
        # self.tower_grads = []

        self._input_states = tf.placeholder("float",
                                            [None, self.RESIZED_SCREEN_X,
                                             self.RESIZED_SCREEN_Y,
                                             self.STATE_FRAMES])
        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None], name="target_Q")
        self._target_input_states = tf.placeholder("float",
                                                   [None, self.RESIZED_SCREEN_X,
                                                    self.RESIZED_SCREEN_Y,
                                                    self.STATE_FRAMES])
        # self.readout_action.append(None)
        # self.cost.append(None)

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
            # _activation_summary(self.output_layer[d])

        self.readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self._action),
                                            axis=1)
        self.cost = tf.reduce_mean(tf.square(self._target - self.readout_action))
        # tf.scalar_summary("loss%d" % d, self.cost)

        # tf.add_to_collection('losses', self.cost)
        # losses = tf.get_collection('losses')
        # total_loss = tf.add_n(losses, name='total_loss')
        grads = self.opt.compute_gradients(self.cost)
        self.tower_grads = grads

        # with tf.device('/gpu:0'):
        with tf.variable_scope('target_conv1'):
            self.target_kernel1 = _variable_with_weight_decay('target_weights',
                                                              shape=[8, 8, self.STATE_FRAMES, 32],
                                                              stddev=0.01,
                                                              wd=0.0)
            self.target_biases1 = _variable_on_gpu('target_biases', [32], tf.constant_initializer(0.01))
            target_conv = tf.nn.conv2d(self._target_input_states, self.target_kernel1, [1, 2, 2, 1],
                                       padding='SAME')
            target_pre_activation = target_conv + self.target_biases1
            target_batch_norm, self.target_beta1, self.target_gamma1 = _batch_normalization(target_pre_activation)
            target_conv1 = tf.nn.relu(target_batch_norm)

        with tf.variable_scope('target_conv2'):
            self.target_kernel2 = _variable_with_weight_decay('target_weights',
                                                              shape=[4, 4, 32, 64],
                                                              stddev=0.01,
                                                              wd=0.0)
            self.target_biases2 = _variable_on_gpu('target_biases', [64], tf.constant_initializer(0.01))
            # conv2
            target_conv = tf.nn.conv2d(target_conv1, self.target_kernel2, [1, 2, 2, 1], padding='SAME')
            target_pre_activation = target_conv + self.target_biases2
            target_batch_norm, self.target_beta2, self.target_gamma2 = _batch_normalization(target_pre_activation)
            target_conv2 = tf.nn.relu(target_batch_norm)

        with tf.variable_scope('target_conv3'):
            self.target_kernel3 = _variable_with_weight_decay('target_weights',
                                                              shape=[3, 3, 64, 64],
                                                              stddev=0.01,
                                                              wd=0.0)
            self.target_biases3 = _variable_on_gpu('target_biases', [64], tf.constant_initializer(0.01))
            # conv3
            target_conv = tf.nn.conv2d(target_conv2, self.target_kernel3, [1, 1, 1, 1], padding='SAME')
            target_pre_activation = target_conv + self.target_biases3
            target_batch_norm, self.target_beta3, self.target_gamma3 = _batch_normalization(target_pre_activation)
            target_conv3 = tf.nn.relu(target_batch_norm)

        with tf.variable_scope('target_local3'):
            self.target_weights4 = _variable_with_weight_decay('target_weights',
                                                               shape=[6400, 256],
                                                               stddev=0.01, wd=0.0)
            self.target_biases4 = _variable_on_gpu('target_biases', [256], tf.constant_initializer(0.01))
            # local3
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(target_conv3, [-1, 6400])
            # dim = reshape.get_shape()[1].value

            target_local3 = tf.nn.relu(tf.matmul(reshape, self.target_weights4) + self.target_biases4)
            # _activation_summary(target_local3)
            # target_fiction_dropout = tf.nn.dropout(target_local3, keep_prob=0.5)

        with tf.variable_scope('target_softmax_linear'):
            self.target_weights6 = _variable_with_weight_decay('target_weights',
                                                               [256, self.ACTIONS_COUNT],
                                                               stddev=0.01, wd=0.0)
            self.target_biases6 = _variable_on_gpu('target_biases', [self.ACTIONS_COUNT],
                                                   tf.constant_initializer(0.01))

            self.target_output_layer = tf.add(tf.matmul(target_local3, self.target_weights6),
                                              self.target_biases6)
        # self._train_operation1 = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(self.cost[0])
        # self._train_operation2 = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(self.cost[1])

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        # grads = average_gradients(self.tower_grads)

        # Apply the gradients to adjust the shared variables.
        self.apply_gradient_op = self.opt.apply_gradients(grads, global_step)

        self._observations = deque()
        self._last_scores = deque()

        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB

        init_op = tf.global_variables_initializer()
        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        # self.merged = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter("/home/uchi/catkin_ws/environment/apple_game/na_logs",
        #                                      self._session.graph)

        self._session.run(init_op)

        self.duration = 0
        self.terminal = True

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        # write into a file
        self.fileqdata = open(self._checkpoint_path + "/Qdata.txt", "w")

        # self.saver = tf.train.Saver()
        # self.saver.restore(self._session, "deep_suna_networks/model40000")

    def game_learning(self):

        screen_array = self.env.getScreenRGB()
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array,
                                                            (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y)),
                                                 cv2.COLOR_BGR2GRAY)
        # screen_resized_grayscaled *= 1/255

        reward = self.env.score()

        if self.terminal:
            print("Steps per episode = ", self.environment.keynum)
            self.env.reset_game()
            self._last_state = np.stack(tuple(screen_resized_grayscaled for _ in range(self.STATE_FRAMES)), axis=2)
        else:
            screen_resized_binary = np.reshape(screen_resized_grayscaled,
                                               (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y, 1))
            current_state = np.append(self._last_state[:, :, 1:], screen_resized_binary, axis=2)
            self._observations.append((self._last_state, self._last_action, reward, current_state, self.terminal))
            # update the old values
            self._last_state = current_state
            # print self._last_state.shape

        self.terminal = self.env.game_over()

        # show resized image

        # cv2.imshow("show", screen_resized_grayscaled)
        # cv2.waitKey(0)

        # print screen_resized_grayscaled
        # set the pixels to all be 0. or 1.
        # _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 1, cv2.THRESH_BINARY)
        # _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

        # # first frame must be handled differently
        # if self._last_state is None:
        #     # the _last_state will contain the image data from the last self.STATE_FRAMES frames
        #     self._last_state = np.stack(tuple(screen_resized_grayscaled for _ in range(self.STATE_FRAMES)), axis=2)
        #     # return self._key_presses_from_action(self._last_action)

        # store the transition in previous_observations

        if len(self._observations) > self.MEMORY_SIZE:
            self._observations.popleft()

        # only train if done observing
        if len(self._observations) > self.OBSERVATION_STEPS:
            start_t_time = time.time()

            self._train()
            self._time += 1
            self.duration += (time.time() - start_t_time)
            if self._time % 10000 == 0:
                train_time = self.duration / 10000
                print ("%.5f per train" % train_time)
                self.duration = 0
        elif len(self._observations) == self.OBSERVATION_STEPS:
            global random_time
            random_time = time.time()

        self._last_action = np.zeros([self.ACTIONS_COUNT])
        new_action = self._choose_next_action(self._last_state)
        self._last_action[new_action] = 1

        # gradually reduce the probability of a random actionself.
        if self._probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB \
                and len(self._observations) > self.OBSERVATION_STEPS:
            self._probability_of_random_action -= \
                (self.INITIAL_RANDOM_ACTION_PROB - self.FINAL_RANDOM_ACTION_PROB) / self.EXPLORE_STEPS

        if self._time % 200 == 0 and self._time != 0:
            global play_steps
            play_steps = self._time
            print("Time: %s random_action_prob: %s reward %s" %
                  (self._time, self._probability_of_random_action, reward))

        if self._time % self.TARGET_NETWORK_UPDATE_FREQ == 0 and self._time > 1:
            # print self._session.run(self.convolution_weights_1)
            change1 = self.target_kernel1.assign(self.kernel1)
            change2 = self.target_kernel2.assign(self.kernel2)
            change3 = self.target_kernel3.assign(self.kernel3)
            change4 = self.target_biases1.assign(self.biases1)
            change5 = self.target_biases2.assign(self.biases2)
            change6 = self.target_biases3.assign(self.biases3)
            change7 = self.target_biases4.assign(self.biases4)
            # change8 = self.target_biases5.assign(self.biases5)
            change9 = self.target_weights4.assign(self.weights4)
            # change10 = self.target_weights5.assign(self.weights5)
            change11 = self.target_weights6.assign(self.weights6)

            self._session.run(
                fetches=[change1, change2, change3, change4, change5,
                         change6, change7, change9, change11])

        self.env.act(self.ple_action_list[new_action])

    def _choose_next_action(self, state):
        # new_action = np.zeros([self.ACTIONS_COUNT])

        if random.random() <= self._probability_of_random_action:
            # choose an action randomly
            action_index = random.randrange(self.ACTIONS_COUNT)
        else:
            self.readout_t = \
                self._session.run(self.output_layer, feed_dict={self._input_states: [state]})[0]

            action_index = np.argmax(self.readout_t)

            q_sum = 0
            for i in range(self.ACTIONS_COUNT):
                q_sum += self.readout_t[i]

            q_average = q_sum / self.ACTIONS_COUNT
            if self._time % 20 == 0:
                print (self.readout_t[action_index])
            if self._time % 200 == 0:
                self.fileqdata.write("%s" % self.readout_t + "\n")

        # new_action[action_index] = 1
        return action_index

    def _train(self):
        # sample a mini_batch to train on
        mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)
        # get the batch variables
        previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
        current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        agents_expected_reward = []
        agents_target_q_reward_per_action = self._session.run(self.target_output_layer,
                                                              feed_dict={self._target_input_states: current_states})

        agents_q_action = self._session.run(self.output_layer,
                                            feed_dict={self._input_states: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.OBS_TERMINAL_INDEX]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                action_number = np.argmax(agents_q_action[i])
                agents_expected_reward.append(
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * agents_target_q_reward_per_action[i][action_number])
                # learn that these actions in these states lead to this reward
        self.the_result = self._session.run(self.apply_gradient_op, feed_dict={
            self._input_states: previous_states,
            self._action: actions,
            self._target: agents_expected_reward})
        '''
        if self._time % 200 == 0:
            self.the_rusult, acc = self._session.run(, feed_dict={self.input_layer: [self._last_state]})
            summary_str = self.the_result[0]
            self.writer.add_summary(summary_str, self._time)
            print("Accuracy at step %s" % self.cost)
            print("Accuracy at step %s" % self._time)
            self.writer.close()
        '''
        # save checkpoints for later
        if self._time % self.SAVE_EVERY_X_STEPS == 0:
            # self._saver.save(self._session, self._checkpoint_path + '/network', global_step=self._time)
            self._saver = tf.train.Saver(var_list=[self.kernel1, self.kernel2, self.kernel3, self.weights4, self.weights6,
                                         self.biases1, self.biases2, self.biases3, self.biases4, self.biases6,
                                                   self.beta1, self.beta2, self.beta3,
                                                   self.gamma1, self.gamma2, self.gamma3])
            save_path = self._saver.save(self._session, self._checkpoint_path + '/model' + str(self._time))
            print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    player = DeepSuna()
    # player.start()
    while True:
        player.game_learning()
