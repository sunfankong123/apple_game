# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames
import os
import random
from collections import deque

import time

from pong_player import PongPlayer
import tensorflow as tf
import numpy as np
import cv2
from pygame.constants import K_DOWN, K_UP, K_LEFT, K_RIGHT


class DeepQPongPlayer(PongPlayer):
    ACTIONS_COUNT = 4  # 3 number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 50000.  # time steps to observe before training
    EXPLORE_STEPS = 1000000.  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.10  # final chance of an action being random
    MEMORY_SIZE = 1000000  # number of observations to remember
    TARGET_NETWORK_UPDATE_FREQ = 10000  # target update frequency
    MINI_BATCH_SIZE = 150  # size of mini batches
    STATE_FRAMES = 4  # number of frames to store in the state
    RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (40, 40)
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
    SAVE_EVERY_X_STEPS = 100000
    LEARN_RATE = 1e-6
    STORE_SCORES_LEN = 200.
    START_TIME = 0
    _session = tf.Session()

    def __init__(self, checkpoint_path="deep_q_pong_networks", playback_mode=False, verbose_logging=False):
        """
        Example of deep q network for pong

        :param checkpoint_path: directory to store checkpoints in
        :type checkpoint_path: str
        :param playback_mode: if true games runs in real time mode and demos itself running
        :type playback_mode: bool
        :param verbose_logging: If true then extra log information is printed to std out
        :type verbose_logging: bool
        """
        self._time = DeepQPongPlayer.START_TIME
        self._playback_mode = playback_mode
        super(DeepQPongPlayer, self).__init__(force_game_fps=8, run_real_time=playback_mode)
        self.verbose_logging = verbose_logging
        self._checkpoint_path = checkpoint_path

        with tf.device('/cpu:0'):
            self.convolution_weights_1 = tf.Variable(
                tf.truncated_normal([8, 8, DeepQPongPlayer.STATE_FRAMES, 32], stddev=0.01))
            self.convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

            self.convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
            self.convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

            self.convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
            self.convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))

            self.feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
            self.feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

            self.feed_forward_weights_2 = tf.Variable(
                tf.truncated_normal([256, DeepQPongPlayer.ACTIONS_COUNT], stddev=0.01))
            self.feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[DeepQPongPlayer.ACTIONS_COUNT]))

            self.target_convolution_weights_1 = tf.Variable(
                tf.truncated_normal([8, 8, DeepQPongPlayer.STATE_FRAMES, 32], stddev=0.01))
            self.target_convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

            self.target_convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
            self.target_convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

            self.target_convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
            self.target_convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))

            self.target_feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
            self.target_feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

            self.target_feed_forward_weights_2 = tf.Variable(
                tf.truncated_normal([256, DeepQPongPlayer.ACTIONS_COUNT], stddev=0.01))
            self.target_feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[DeepQPongPlayer.ACTIONS_COUNT]))

        # network weights
        self.input_layer = tf.placeholder("float",
                                          [None, DeepQPongPlayer.RESIZED_SCREEN_X, DeepQPongPlayer.RESIZED_SCREEN_Y,
                                           DeepQPongPlayer.STATE_FRAMES])

        hidden_convolutional_layer_1 = tf.nn.relu(
            tf.nn.conv2d(self.input_layer, self.convolution_weights_1, strides=[1, 2, 2, 1],
                         padding="SAME") + self.convolution_bias_1)

        hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_2 = tf.nn.relu(
            tf.nn.conv2d(hidden_max_pooling_layer_1, self.convolution_weights_2, strides=[1, 2, 2, 1],
                         padding="SAME") + self.convolution_bias_2)

        hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3 = tf.nn.relu(
            tf.nn.conv2d(hidden_max_pooling_layer_2, self.convolution_weights_3,
                         strides=[1, 1, 1, 1], padding="SAME") + self.convolution_bias_3)

        hidden_max_pooling_layer_3 = tf.nn.max_pool(hidden_convolutional_layer_3, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_3, [-1, 256])

        final_hidden_activations = tf.nn.relu(
            tf.matmul(hidden_convolutional_layer_3_flat, self.feed_forward_weights_1) + self.feed_forward_bias_1)

        fiction_dropout = tf.nn.dropout(final_hidden_activations, keep_prob=0.5)

        self.output_layer = tf.matmul(fiction_dropout, self.feed_forward_weights_2) + self.feed_forward_bias_2

        #    return input_layer, output_layer

        # @staticmethod
        # def _create_target_network():
        # network weights
        self.target_input_layer = tf.placeholder("float",
                                                 [None, DeepQPongPlayer.RESIZED_SCREEN_X,
                                                  DeepQPongPlayer.RESIZED_SCREEN_Y,
                                                  DeepQPongPlayer.STATE_FRAMES])

        target_hidden_convolutional_layer_1 = tf.nn.relu(
            tf.nn.conv2d(self.target_input_layer, self.target_convolution_weights_1, strides=[1, 2, 2, 1],
                         padding="SAME") + self.target_convolution_bias_1)

        target_hidden_max_pooling_layer_1 = tf.nn.max_pool(target_hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                                           strides=[1, 2, 2, 1], padding="SAME")

        target_hidden_convolutional_layer_2 = tf.nn.relu(
            tf.nn.conv2d(target_hidden_max_pooling_layer_1, self.target_convolution_weights_2, strides=[1, 2, 2, 1],
                         padding="SAME") + self.target_convolution_bias_2)

        target_hidden_max_pooling_layer_2 = tf.nn.max_pool(target_hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                                                           strides=[1, 2, 2, 1], padding="SAME")

        target_hidden_convolutional_layer_3 = tf.nn.relu(
            tf.nn.conv2d(target_hidden_max_pooling_layer_2, self.target_convolution_weights_3,
                         strides=[1, 1, 1, 1], padding="SAME") + self.target_convolution_bias_3)

        target_hidden_max_pooling_layer_3 = tf.nn.max_pool(target_hidden_convolutional_layer_3, ksize=[1, 2, 2, 1],
                                                           strides=[1, 2, 2, 1], padding="SAME")

        target_hidden_convolutional_layer_3_flat = tf.reshape(target_hidden_max_pooling_layer_3, [-1, 256])

        target_final_hidden_activations = tf.nn.relu(
            tf.matmul(target_hidden_convolutional_layer_3_flat,
                      self.target_feed_forward_weights_1) + self.target_feed_forward_bias_1)

        self.target_output_layer = tf.matmul(target_final_hidden_activations,
                                             self.target_feed_forward_weights_2) + self.target_feed_forward_bias_2

        # self._input_layer, self._output_layer, self._target_input_layer,
        # self._target_output_layer = self._create_network(self._time)
        # self._target_input_layer, self._target_output_layer = DeepQPongPlayer._create_target_network()

        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None], name="target_Q")

        with tf.name_scope("loss"):
            readout_action = tf.reduce_sum(tf.mul(self.output_layer, self._action), reduction_indices=1)
            self.cost = tf.reduce_mean(tf.square(self._target - readout_action))

        with tf.name_scope("train"):
            self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(self.cost)

        # Merge all the summaries and write them out to /na_logs
        # self.merged = tf.merge_all()
        # self.writer = tf.train.SummaryWriter("/home/uchi/catkin_ws/src/PyGamePlayer/apple_game/na_logs",
        # self._session.graph)

        self._observations = deque()
        self._last_scores = deque()

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = None
        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB

        self._session.run(tf.global_variables_initializer())

        self.duration = 0
        # write into a file
        self.fileqdata = open("Qdata.txt", "w")
        # self.fileinputdata = open("inputdata.txt", "r")
        #   self.fileloss = open("lossdata.txt", "w")

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)

        # self.saver = tf.train.Saver()
        # self.saver.restore(self._session, "networks_middle_presentation/model7300000")
        '''
        self._saver.restore(self._session, "deep_q_pong_networks/network-1040000")

        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif playback_mode:
            raise Exception("Could not load checkpoints for playback")
        '''

    def get_keys_pressed(self, screen_array, reward, terminal):
        # scale down screen image
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array,
                                                            (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y)),
                                                 cv2.COLOR_BGR2GRAY)

        # show resized image

        # cv2.imshow("show", screen_resized_grayscaled)
        # cv2.waitKey(0)

        # print screen_resized_grayscaled
        # set the pixels to all be 0. or 1.
        # _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 1, cv2.THRESH_BINARY)
        # _, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

        if reward != 0.0:
            self._last_scores.append(reward)
            if len(self._last_scores) > self.STORE_SCORES_LEN:
                self._last_scores.popleft()

        # first frame must be handled differently
        if self._last_state is None:
            # the _last_state will contain the image data from the last self.STATE_FRAMES frames
            self._last_state = np.stack(tuple(screen_resized_grayscaled for _ in range(self.STATE_FRAMES)), axis=2)
            return DeepQPongPlayer._key_presses_from_action(self._last_action)

        screen_resized_binary = np.reshape(screen_resized_grayscaled,
                                           (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y, 1))
        current_state = np.append(self._last_state[:, :, 1:], screen_resized_binary, axis=2)

        if not self._playback_mode:
            # store the transition in previous_observations
            self._observations.append((self._last_state, self._last_action, reward, current_state, terminal))

            if len(self._observations) > self.MEMORY_SIZE:
                self._observations.popleft()

            # only train if done observing
            if len(self._observations) > self.OBSERVATION_STEPS:
                # start_time = time.time()

                self._train()
                self._time += 1
                # self.duration += (time.time() - start_time)
                # if self._time % 10 == 0:
                #     average = self.duration / self._time
                #     print ("%.5f per train" % average)

        # update the old values
        self._last_state = current_state

        self._last_action = self._choose_next_action()

        if not self._playback_mode:
            # gradually reduce the probability of a random actionself.
            if self._probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB \
                    and len(self._observations) > self.OBSERVATION_STEPS:
                self._probability_of_random_action -= \
                    (self.INITIAL_RANDOM_ACTION_PROB - self.FINAL_RANDOM_ACTION_PROB) / self.EXPLORE_STEPS

            if self._time % 100 == 0 and self._time != 0:
                print("Time: %s random_action_prob: %s reward %s" %
                      (self._time, self._probability_of_random_action, reward))

        if self._time % DeepQPongPlayer.TARGET_NETWORK_UPDATE_FREQ == 0 and self._time != 1:
            # print self._session.run(self.convolution_weights_1)
            change1 = self.target_convolution_weights_1.assign(self.convolution_weights_1)
            # print self._session.run(self.convolution_weights_1)
            change2 = self.target_convolution_weights_2.assign(self.convolution_weights_2)
            change3 = self.target_convolution_weights_3.assign(self.convolution_weights_3)
            change4 = self.target_convolution_bias_1.assign(self.convolution_bias_1)
            change5 = self.target_convolution_bias_2.assign(self.convolution_bias_2)
            change6 = self.target_convolution_bias_3.assign(self.convolution_bias_3)
            change7 = self.target_feed_forward_weights_1.assign(self.feed_forward_weights_1)
            change8 = self.target_feed_forward_weights_2.assign(self.feed_forward_weights_2)
            change9 = self.target_feed_forward_bias_1.assign(self.feed_forward_bias_1)
            change10 = self.target_feed_forward_bias_2.assign(self.feed_forward_bias_2)

            self._session.run(
                fetches=[change1, change2, change3, change4, change5, change6, change7, change8, change9, change10])

        return DeepQPongPlayer._key_presses_from_action(self._last_action)

    def _choose_next_action(self):
        new_action = np.zeros([self.ACTIONS_COUNT])

        if (not self._playback_mode) and (random.random() <= self._probability_of_random_action):
            # choose an action randomly
            action_index = random.randrange(self.ACTIONS_COUNT)
        else:
            # choose an action given our last state

            self.readout_t = self._session.run(self.output_layer, feed_dict={self.input_layer: [self._last_state]})[0]
            # tf.scalar_summary("Q values", self.readout_t)
            # if self.verbose_logging:
            # print("Action Q-Values are %s" % self.readout_t)

            action_index = np.argmax(self.readout_t)

            q_sum = 0
            for i in range(DeepQPongPlayer.ACTIONS_COUNT):
                q_sum += self.readout_t[i]

            q_average = q_sum / DeepQPongPlayer.ACTIONS_COUNT
            if self._time % 10 == 0:
                print self.readout_t[action_index]
            if self._time % 200 == 0:
                self.fileqdata.write("%s" % q_average + "\n")

        new_action[action_index] = 1
        return new_action

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
                                                              feed_dict={self.target_input_layer: current_states})

        agents_q_action = self._session.run(self.output_layer,
                                            feed_dict={self.input_layer: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.OBS_TERMINAL_INDEX]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                action_number = np.argmax(agents_q_action[i])
                agents_expected_reward.append(
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * agents_target_q_reward_per_action[i][action_number])
                # learn that these actions in these states lead to this reward
        self.the_result = self._session.run(self._train_operation, feed_dict={
            self.input_layer: previous_states,
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
            self._saver = tf.train.Saver()
            save_path = self._saver.save(self._session, self._checkpoint_path + '/model' + str(self._time))
            print("Model saved in file: %s" % save_path)

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

    '''
    @staticmethod
    def _target_network_update():
        target_network = DeepQPongPlayer._create_network()
        q_network = DeepQPongPlayer._create_network()
        print DeepQPongPlayer.time
        target_network.target_convolution_weights_1.assign(q_network.convolution_weights_1)
        target_network.target_convolution_weights_2.assign(q_network.convolution_weights_2)
        target_network.target_convolution_weights_3.assign(q_network.convolution_weights_3)
        target_network.target_convolution_bias_1.assign(q_network.convolution_bias_1)
        target_network.target_convolution_bias_2.assign(q_network.convolution_bias_2)
        target_network.target_convolution_bias_3.assign(q_network.convolution_bias_3)
        target_network.target_feed_forward_weights_1.assign(q_network.feed_forward_weights_1)
        target_network.target_feed_forward_weights_2.assign(q_network.feed_forward_weights_2)
        target_network.target_feed_forward_bias_1.assign(q_network.feed_forward_bias_1)
        target_network.target_feed_forward_bias_2.assign(q_network.feed_forward_bias_2)
    '''


if __name__ == '__main__':
    player = DeepQPongPlayer()
    player.start()
