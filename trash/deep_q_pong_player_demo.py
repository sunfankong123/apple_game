# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames
import os
import random
from collections import deque
from pong_player import PongPlayer
import tensorflow as tf
import numpy as np
import cv2
from pygame.constants import K_DOWN, K_UP, K_LEFT, K_RIGHT


class DeepQPongPlayer(PongPlayer):
    ACTIONS_COUNT = 4  # 3 number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.80  # decay rate of past observations
    OBSERVATION_STEPS = 50000.  # time steps to observe before training
    EXPLORE_STEPS = 5000000.  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0  # final chance of an action being random
    MEMORY_SIZE = 500000  # number of observations to remember
    TARGET_NETWORK_UPDATE_FREQ = 5000  # target update frequency
    MINI_BATCH_SIZE = 150  # size of mini batches
    STATE_FRAMES = 4  # number of frames to store in the state
    RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (40, 40)
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
    SAVE_EVERY_X_STEPS = 100000
    LEARN_RATE = 1e-6
    STORE_SCORES_LEN = 200.
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
        self._time = 0
        self._playback_mode = playback_mode
        super(DeepQPongPlayer, self).__init__(force_game_fps=8, run_real_time=playback_mode)
        self.verbose_logging = verbose_logging
        self._checkpoint_path = checkpoint_path

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

        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None], name="target_Q")

        with tf.name_scope("loss"):
            readout_action = tf.reduce_sum(tf.mul(self.output_layer, self._action), reduction_indices=1)
            self.cost = tf.reduce_mean(tf.square(self._target - readout_action))
            # tf.scalar_summary("loss", self.cost)

        with tf.name_scope("train"):
            self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(self.cost)

        self._observations = deque()
        self._last_scores = deque()

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = None
        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB

        self._session.run(tf.initialize_all_variables())

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)

        self._saver = tf.train.Saver()
        # self._saver.restore(self._session, "networks_bootstrap_k3/model2600000")

    def get_keys_pressed(self, screen_array, reward, terminal):
        # scale down screen image
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array,
                                                            (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y)),
                                                 cv2.COLOR_BGR2GRAY)
        filename = "imagedata/image" + str(self._time) + ".png"
        cv2.imwrite(filename, screen_resized_grayscaled)
        self._time += 1

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

        # update the old values
        self._last_state = current_state

        self._last_action = self._choose_next_action()

        return DeepQPongPlayer._key_presses_from_action(self._last_action)

    def _choose_next_action(self):
        new_action = np.zeros([self.ACTIONS_COUNT])
        action_index = int(raw_input('Enter your input:'))

        # self.readout_t = self._session.run(self.output_layer, feed_dict={self.input_layer: [self._last_state]})[0]

        # action_index = np.argmax(self.readout_t)

        new_action[action_index] = 1
        return new_action

    @staticmethod
    def _key_presses_from_action(action_set):
        if action_set[0] == 1:
            return [K_DOWN]
        # elif action_set[1] == 1:
        #    return []
        elif action_set[1] == 1:
            return [K_UP]
        elif action_set[2] == 1:
            return [K_LEFT]
        elif action_set[3] == 1:
            return [K_RIGHT]

        raise Exception("Unexpected action")

if __name__ == '__main__':
    player = DeepQPongPlayer()
    player.start()
