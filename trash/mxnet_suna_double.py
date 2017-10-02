# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Software License Agreement (BSD License)
#
# Copyright (c) 2017, SHIBAURA INSTUTUE OF TECHNOLOGY
# All rights reserved.
#

import random
from collections import deque

import time
import os

from ai_player import AI_Player
import cv2
from pygame.constants import K_DOWN, K_UP, K_LEFT, K_RIGHT

import copy
import numpy as np
import cupy
# from chainer import cuda, Chain, Variable, optimizers, serializers
# from chainer import links
# import chainer.functions as funcitons

import mxnet as mx
from mxnet import nd


class ChainerDQNclass:
    STATE_FRAMES = 4  # number of frames to store in the state

    def __init__(self):
        self.num_of_actions = 4

        print "Initializing DQN..."

        print "Model Building"
        self.model = Chain(
            l1=links.Convolution2D(self.STATE_FRAMES, 32, ksize=8, stride=4, nobias=False, wscale=0.01),
            l2=links.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=0.01),
            l3=links.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=0.01),
            l4=links.Linear(3136, 512, wscale=0.01),
            q_value=links.Linear(512, self.num_of_actions)
        ).to_gpu()

        self.model_target = copy.deepcopy(self.model)

        # self.optimizer = optimizers.Adam(alpha=1e-6)
        # self.optimizer.use_cleargrads()
        # self.optimizer.setup(self.model)

    def Q_func(self, state):
        h1 = funcitons.relu(self.model.l1(state))  # scale inputs in [0.0 1.0]
        h2 = funcitons.relu(self.model.l2(h1))
        h3 = funcitons.relu(self.model.l3(h2))
        h4 = funcitons.relu(self.model.l4(h3))
        Q = self.model.q_value(h4)
        return Q

    def Q_func_target(self, state):
        h1 = funcitons.relu(self.model_target.l1(state))  # scale inputs in [0.0 1.0]
        h2 = funcitons.relu(self.model_target.l2(h1))
        h3 = funcitons.relu(self.model_target.l3(h2))
        h4 = funcitons.relu(self.model_target.l4(h3))
        Q = self.model_target.q_value(h4)
        return Q

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)


class DeepSuna(AI_Player):
    ACTIONS_COUNT = 4  # 3 number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 5000.  # time steps to observe before training
    EXPLORE_STEPS = 1000000.  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.10  # final chance of an action being random
    MEMORY_SIZE = 1000000  # number of observations to remember
    TARGET_NETWORK_UPDATE_FREQ = 10000  # target update frequency
    MINI_BATCH_SIZE = 50  # size of mini batches
    RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (84, 84)
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
    SAVE_EVERY_X_STEPS = 100000
    LEARN_RATE = 1e-6
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
    DECAY_STEPS = 200000
    STORE_SCORES_LEN = 200.
    START_TIME = 0

    def __init__(self, checkpoint_path="deep_suna_networks", playback_mode=False, verbose_logging=False):
        """
        Example of deep q network for pong

        :param checkpoint_path: directory to store checkpoints in
        :type checkpoint_path: str
        :param playback_mode: if true games runs in real time mode and demos itself running
        :type playback_mode: bool
        :param verbose_logging: If true then extra log information is printed to std out
        :type verbose_logging: bool
        """

        self._time = self.START_TIME
        self._playback_mode = playback_mode
        super(DeepSuna, self).__init__(force_game_fps=8, run_real_time=playback_mode)
        self.verbose_logging = verbose_logging
        self._checkpoint_path = checkpoint_path

        self._observations = deque()
        self.duration = 0
        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB

        self.chainer_dqn_class = ChainerDQNclass()

        print "Initializing Optimizer"
        self.optimizer = optimizers.Adam(alpha=self.LEARN_RATE)
        self.optimizer.use_cleargrads()
        self.optimizer.setup(self.chainer_dqn_class.model)

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        self.fileqdata = open(self._checkpoint_path + "/Qdata.txt", "w")
        # set the first action to do nothing
        self._last_action = nd.zeros((self.ACTIONS_COUNT,))
        self._last_action[1] = 1
        self._last_state = nd.zeros((self.chainer_dqn_class.STATE_FRAMES,
                                     self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y), dtype=np.uint8)
        self.first_frame = True

    def get_keys_pressed(self, screen_array, reward, terminal, turn):
        # scale down screen image
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array,
                                                            (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y)),
                                                 cv2.COLOR_BGR2GRAY)

        # cv2.imshow("show", screen_resized_grayscaled)
        # cv2.waitKey(0)

        # print screen_resized_grayscaled
        # set the pixels to all be 0. or 1.
        # _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 1, cv2.THRESH_BINARY)
        # _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

        # first frame must be handled differently
        if self.first_frame is True:
            self._last_state[0] = screen_resized_grayscaled
            compute_state = nd.array(self._last_state.reshape(1, self.chainer_dqn_class.STATE_FRAMES,
                                                              self.RESIZED_SCREEN_X,
                                                              self.RESIZED_SCREEN_Y))
            self.first_frame = False
            return self._key_presses_from_action(self._choose_next_action(compute_state))

        current_state = np.asanyarray([self._last_state[1], self._last_state[2], self._last_state[3],
                                       screen_resized_grayscaled], dtype=np.uint8)
        compute_state = nd.array(current_state.reshape(1, self.chainer_dqn_class.STATE_FRAMES,
                                                       self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y))

        if not self._playback_mode:
            # store the transition in previous_observations
            self._observations.append((self._last_state, self._last_action, reward, current_state, terminal))

            if len(self._observations) > self.MEMORY_SIZE:
                self._observations.popleft()

            # only train if done observing
            if len(self._observations) > self.OBSERVATION_STEPS:
                start_time = time.time()

                self._train()
                self._time += 1
                self.duration += (time.time() - start_time)
                if self._time % 100 == 0:
                    average = self.duration / self._time
                    print ("%.5f per train" % average)

        # update the old values
        self._last_state = current_state

        self._last_action = self._choose_next_action(compute_state)

        if not self._playback_mode:
            # gradually reduce the probability of a random actionself.
            if self._probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB \
                    and len(self._observations) > self.OBSERVATION_STEPS:
                self._probability_of_random_action -= \
                    (self.INITIAL_RANDOM_ACTION_PROB - self.FINAL_RANDOM_ACTION_PROB) / self.EXPLORE_STEPS

            if self._time % 100 == 0 and self._time != 0:
                # summary_str = self._session.run(self.merged)
                # self.writer.add_summary(summary_str, self._time)
                print("Time: %s random_action_prob: %s reward %s" %
                      (self._time, self._probability_of_random_action, reward))

        if self._time % self.TARGET_NETWORK_UPDATE_FREQ == 0 and self._time > 1:
            self.chainer_dqn_class.target_model_update()

        return self._key_presses_from_action(self._last_action)

    def _choose_next_action(self, state):
        new_action = nd.zeros((self.ACTIONS_COUNT,))

        if (not self._playback_mode) and (random.random() <= self._probability_of_random_action):
            # choose an action randomly
            action_index = random.randrange(self.ACTIONS_COUNT)
        else:
            # choose an action given our last state

            # self._input_states = self._last_state
            self.readout_t = self.chainer_dqn_class.Q_func(state)
            self.readout_t = self.readout_t.data

            action_index = np.argmax(self.readout_t[0])

            q_sum = 0
            for i in range(self.ACTIONS_COUNT):
                q_sum += self.readout_t[0][i]

            q_average = q_sum / self.ACTIONS_COUNT
            if self._time % 10 == 0:
                print self.readout_t[0][action_index]
            if self._time % 200 == 0:
                self.fileqdata.write("%s" % q_average + "\n")

        new_action[action_index] = 1
        return new_action

    def _train(self):
        # sample a mini_batch to train on
        mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)
        previous_states = nd.array((self.MINI_BATCH_SIZE, 4, self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y))
        actions = nd.array((self.MINI_BATCH_SIZE, 4))
        rewards = nd.array((self.MINI_BATCH_SIZE, 1))
        current_states = nd.array((self.MINI_BATCH_SIZE, 4, self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y))
        # get the batch variables
        for i in xrange(self.MINI_BATCH_SIZE):
            previous_states[i] = nd.array(mini_batch[i][0])
            actions[i] = nd.array(mini_batch[i][1])
            rewards[i] = nd.array(mini_batch[i][2])
            current_states[i] = nd.array(mini_batch[i][3])

        previous_states = Variable(previous_states)
        current_states = Variable(current_states)
        # agents_expected_reward = []
        agents_target_q_reward_per_action = self.chainer_dqn_class.Q_func_target(current_states)
        # print agents_target_q_reward_per_action.data
        tmp = list(agents_target_q_reward_per_action.data.get())
        agents_target_q_reward_per_action = nd.array(tmp)
        # agents_target_q_reward_per_action = agents_target_q_reward_per_action.data

        agents_q_action = self.chainer_dqn_class.Q_func(current_states)

        agents_q_action = agents_q_action.data
        # print agents_q_action
        # print actions

        previous_q = self.chainer_dqn_class.Q_func(previous_states)
        agents_expected_reward = np.asanyarray(previous_q.data.get(), dtype=np.float32)

        for i in range(len(mini_batch)):
            action_number = np.argmax(agents_q_action[i].get())
            if mini_batch[i][self.OBS_TERMINAL_INDEX]:
                # this was a terminal frame so there is no future reward...
                temp = rewards[i]
            else:
                temp = \
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * agents_target_q_reward_per_action[i][action_number]
                # learn that these actions in these states lead to this reward
            agents_expected_reward[i, action_number] = temp

        '''

        td = Variable(cuda.to_gpu(agents_expected_reward)) - previous_q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td / abs(td_tmp) * (abs(td.data) > 1)

        zero_array = cupy.zeros((self.MINI_BATCH_SIZE, self.ACTIONS_COUNT), dtype=cupy.float32)
        zero_val = Variable(zero_array)
        self.chainer_dqn_class.model.cleargrads()
        loss = funcitons.mean_squared_error(td_clip, zero_val)
        loss.backward()
        self.optimizer.update()
        '''

        # save checkpoints for later
        if self._time % self.SAVE_EVERY_X_STEPS == 0:
            print "model saved"
            mx.sym.save(self._checkpoint_path + '/model' + str(self._time))

    @staticmethod
    def _key_presses_from_action(action_set):
        if action_set[0] == 1:
            return [K_DOWN]
        # elif action_set[1] == 1:
        #     return []
        elif action_set[1] == 1:
            return [K_UP]
        elif action_set[2] == 1:
            return [K_LEFT]
        elif action_set[3] == 1:
            return [K_RIGHT]

        raise Exception("Unexpected action")


if __name__ == '__main__':
    player = DeepSuna()
    player.start()
