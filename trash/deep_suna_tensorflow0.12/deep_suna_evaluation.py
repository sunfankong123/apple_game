# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames
import re
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from pygame.constants import K_DOWN, K_UP, K_LEFT, K_RIGHT

from ai_player import AI_Player


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
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
    with tf.device('/cpu:0'):
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
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

_turn = 0


class DeepSuna(AI_Player):
    ACTIONS_COUNT = 4  # 3 number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 50000.  # time steps to observe before training
    EXPLORE_STEPS = 1000000.  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 0.05  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.0  # final chance of an action being random
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

        tf.reset_default_graph()
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
        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None], name="target_Q")
        self._target_input_states = tf.placeholder("float",
                                                   [None, self.RESIZED_SCREEN_X,
                                                    self.RESIZED_SCREEN_Y,
                                                    self.STATE_FRAMES])
        # Calculate the gradients for each model tower.
        self.tower_grads = []

        with tf.device('/gpu:0'):
            with tf.variable_scope('conv1'):
                self.kernel1 = _variable_with_weight_decay('weights',
                                                           shape=[8, 8, self.STATE_FRAMES, 32],
                                                           stddev=0.01,
                                                           wd=None)
                self.biases1 = _variable_on_cpu('biases', [32], tf.constant_initializer(0.01))
                conv = tf.nn.conv2d(self._input_states, self.kernel1, [1, 2, 2, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, self.biases1)
                conv1 = tf.nn.relu(pre_activation)
            # _activation_summary(conv1)

            # pool1
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            # norm1
            # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            #                   name='norm1')

            with tf.variable_scope('conv2'):
                self.kernel2 = _variable_with_weight_decay('weights',
                                                           shape=[4, 4, 32, 64],
                                                           stddev=0.01,
                                                           wd=0.0)
                self.biases2 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.01))
                # conv2
                conv = tf.nn.conv2d(pool1, self.kernel2, [1, 2, 2, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, self.biases2)
                conv2 = tf.nn.relu(pre_activation)
                # _activation_summary(conv2)

            # norm2
            # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            #                   name='norm2')
            # pool2
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            with tf.variable_scope('conv3'):
                self.kernel3 = _variable_with_weight_decay('weights',
                                                           shape=[3, 3, 64, 64],
                                                           stddev=0.01,
                                                           wd=0.0)
                self.biases3 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.01))
                # conv3
                conv = tf.nn.conv2d(pool2, self.kernel3, [1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, self.biases3)
                conv3 = tf.nn.relu(pre_activation)
                # _activation_summary(conv3)
            # pool3
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            with tf.variable_scope('local3'):
                self.weights4 = _variable_with_weight_decay('weights', shape=[256, 256],
                                                            stddev=0.01, wd=0.0)
                self.biases4 = _variable_on_cpu('biases', [256], tf.constant_initializer(0.01))
                # local3
                # Move everything into depth so we can perform a single matrix multiply.
                reshape = tf.reshape(pool3, [-1, 256])
                # dim = reshape.get_shape()[1].value
                local3 = tf.nn.relu(tf.matmul(reshape, self.weights4) + self.biases4)
                # _activation_summary(local3)
            # fiction_dropout = tf.nn.dropout(local3, keep_prob=0.5)
            '''
            # local4
            with tf.variable_scope('local4') as scope:
                self.weights5 = _variable_with_weight_decay('weights', shape=[384, 256],
                                                            stddev=0.04, wd=0.004)
                self.biases5 = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
                local4 = tf.nn.relu(tf.matmul(local3, self.weights5) + self.biases5, name=scope.name)
                _activation_summary(local4)
            '''

            with tf.variable_scope('softmax_linear'):
                self.weights6 = _variable_with_weight_decay('weights', [256, self.ACTIONS_COUNT],
                                                            stddev=0.01, wd=0.0)
                self.biases6 = _variable_on_cpu('biases', [self.ACTIONS_COUNT],
                                                tf.constant_initializer(0.01))
                self.output_layer = tf.add(tf.matmul(local3, self.weights6), self.biases6)

        with tf.device('/gpu:0'):
            with tf.variable_scope('target_conv1'):
                self.target_kernel1 = _variable_with_weight_decay('target_weights',
                                                                  shape=[8, 8, self.STATE_FRAMES, 32],
                                                                  stddev=0.01,
                                                                  wd=0.0)
                self.target_biases1 = _variable_on_cpu('target_biases', [32], tf.constant_initializer(0.01))
                target_conv = tf.nn.conv2d(self._target_input_states, self.target_kernel1, [1, 2, 2, 1],
                                           padding='SAME')
                target_pre_activation = target_conv + self.target_biases1
                target_conv1 = tf.nn.relu(target_pre_activation)
                # _activation_summary(target_conv1)

            target_pool1 = tf.nn.max_pool(target_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                          padding='SAME', name='target_pool1')

            with tf.variable_scope('target_conv2'):
                self.target_kernel2 = _variable_with_weight_decay('target_weights',
                                                                  shape=[4, 4, 32, 64],
                                                                  stddev=0.01,
                                                                  wd=0.0)
                self.target_biases2 = _variable_on_cpu('target_biases', [64], tf.constant_initializer(0.01))
                # conv2
                target_conv = tf.nn.conv2d(target_pool1, self.target_kernel2, [1, 2, 2, 1], padding='SAME')
                target_pre_activation = target_conv + self.target_biases2
                target_conv2 = tf.nn.relu(target_pre_activation)
                # _activation_summary(target_conv2)

            target_pool2 = tf.nn.max_pool(target_conv2, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME', name='target_pool2')

            with tf.variable_scope('target_conv3'):
                self.target_kernel3 = _variable_with_weight_decay('target_weights',
                                                                  shape=[3, 3, 64, 64],
                                                                  stddev=0.01,
                                                                  wd=0.0)
                self.target_biases3 = _variable_on_cpu('target_biases', [64], tf.constant_initializer(0.01))
                # conv3
                target_conv = tf.nn.conv2d(target_pool2, self.target_kernel3, [1, 1, 1, 1], padding='SAME')
                target_pre_activation = target_conv + self.target_biases3
                target_conv3 = tf.nn.relu(target_pre_activation)
                # _activation_summary(target_conv3)

            target_pool3 = tf.nn.max_pool(target_conv3, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME', name='target_pool2')

            with tf.variable_scope('target_local3'):
                self.target_weights4 = _variable_with_weight_decay('target_weights',
                                                                   shape=[256, 256],
                                                                   stddev=0.01, wd=0.0)
                self.target_biases4 = _variable_on_cpu('target_biases', [256], tf.constant_initializer(0.01))
                # local3
                # Move everything into depth so we can perform a single matrix multiply.
                reshape = tf.reshape(target_pool3, [-1, 256])
                # dim = reshape.get_shape()[1].value

                target_local3 = tf.nn.relu(tf.matmul(reshape, self.target_weights4) + self.target_biases4)
                # _activation_summary(target_local3)
            # target_fiction_dropout = tf.nn.dropout(target_local3, keep_prob=0.5)

            with tf.variable_scope('target_softmax_linear'):
                self.target_weights6 = _variable_with_weight_decay('target_weights',
                                                                   [256, self.ACTIONS_COUNT],
                                                                   stddev=0.01, wd=0.0)
                self.target_biases6 = _variable_on_cpu('target_biases', [self.ACTIONS_COUNT],
                                                       tf.constant_initializer(0.01))

                self.target_output_layer = tf.add(tf.matmul(target_local3, self.target_weights6),
                                                  self.target_biases6)

        self._observations = deque()
        self._last_scores = deque()

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        self.duration = 0
        self.count = 1
        self.restore_filename = 1000000

        self.saver = tf.train.Saver()
        # self.saver.restore(self._session, "networks_middle_presentation/model7300000")

        self.saver.restore(self._session, "networks_pinkjuki_no_dropout_170205/model"+str(self.restore_filename))

        '''
        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif playback_mode:
            raise Exception("Could not load checkpoints for playback")
        '''

    def get_keys_pressed(self, screen_array, reward, terminal, turn):
        # scale down screen image
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array,
                                                            (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y)),
                                                 cv2.COLOR_BGR2GRAY)
        self._time += 1

        # global _turn
        # _turn = turn
        # self.count = 1
        # print turn
        # print self.count
        if turn == self.count:
            # player.stop()
            # print "aaa"
            self.restore_filename += 500000
            self.saver.restore(self._session, "networks_pinkjuki_no_dropout_170205/model"+str(self.restore_filename))
            self.count += 1
            # player.start()

        if reward != 0.0:
            self._last_scores.append(reward)
            if len(self._last_scores) > self.STORE_SCORES_LEN:
                self._last_scores.popleft()

        # first frame must be handled differently
        if self._last_state is None:
            # the _last_state will contain the image data from the last self.STATE_FRAMES frames
            self._last_state = np.stack(tuple(screen_resized_grayscaled for _ in range(self.STATE_FRAMES)), axis=2)
            return self._key_presses_from_action(self._last_action)

        screen_resized_binary = np.reshape(screen_resized_grayscaled,
                                           (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y, 1))
        current_state = np.append(self._last_state[:, :, 1:], screen_resized_binary, axis=2)

        # update the old values
        self._last_state = current_state

        self._last_action = self._choose_next_action()

        return self._key_presses_from_action(self._last_action)

    def _choose_next_action(self):
        new_action = np.zeros([self.ACTIONS_COUNT])
        # action_index = int(raw_input('Enter your input:'))

        self.readout_t \
            = self._session.run(self.output_layer, feed_dict={self._input_states: [self._last_state]})[0]

        action_index = np.argmax(self.readout_t)

        new_action[action_index] = 1
        return new_action

    '''
    def _choose_next_action(self):
        new_action = np.zeros([self.ACTIONS_COUNT])

        if (not self._playback_mode) and (random.random() <= self._probability_of_random_action):
            # choose an action randomly
            action_index = random.randrange(self.ACTIONS_COUNT)
        else:
            # choose an action given our last state

            # self._input_states = self._last_state
            self.readout_t = \
                self._session.run(self.output_layer[0], feed_dict={self._input_states[0]: [self._last_state]})[0]
            # tf.scalar_summary("Q values", self.readout_t)
            # if self.verbose_logging:
            # print("Action Q-Values are %s" % self.readout_t)

            action_index = np.argmax(self.readout_t)

        new_action[action_index] = 1
        return new_action
    '''

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

