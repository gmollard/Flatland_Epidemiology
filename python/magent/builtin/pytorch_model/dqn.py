import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class replaymemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, env, handle, name,
                 batch_size=64, learning_rate=1e-4, reward_decay=0.999999,
                 train_freq=1, target_update=2000, memory_size=2 ** 20, eval_obs=None,
                 use_dueling=True, use_double=True, use_conv=True,
                 custom_view_space=None, custom_feature_space=None,
                 num_gpu=1, infer_batch_size=8192, network_type=0):

        self.env = env
        self.handle = handle
        self.view_space = env.get_view_space(handle)

        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_freq = train_freq  # train time of every sample (s,a,r,s') TODO: understand what it does
        self.target_update = target_update  # target network update frequency TODO: understand
        self.eval_obs = eval_obs
        self.infer_batch_size = infer_batch_size  # maximum batch size when infer actions,
        # change this to fit your GPU memory if you meet a OOM

        self.use_dueling = use_dueling
        self.use_double = use_double
        self.num_gpu = num_gpu
        self.network_type = network_type

        self.train_ct = 0

        # loss
        self.gamma = reward_decay
        # self.actions_onehot = tf.one_hot(self.action, self.num_actions)
        # td_error = tf.square(self.target - tf.reduce_sum(tf.multiply(self.actions_onehot, self.qvalues), axis=1))
        # self.loss = tf.reduce_sum(td_error * self.mask) / tf.reduce_sum(self.mask)
        #
        # # train op (clip gradient)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # self.train_op = optimizer.apply_gradients(zip(gradients, variables))
        #
        # # output action
        # def out_action(qvalues):
        #     best_action = tf.argmax(qvalues, axis=1)
        #     best_action = tf.to_int32(best_action)
        #     random_action = tf.random_uniform(tf.shape(best_action), 0, self.num_actions, tf.int32)
        #     should_explore = tf.random_uniform(tf.shape(best_action), 0, 1) < self.eps
        #     return tf.where(should_explore, random_action, best_action)
        #
        # self.output_action = out_action(self.qvalues)
        # if self.num_gpu > 1:
        #     self.infer_out_action = [out_action(qvalue) for qvalue in self.infer_qvalues]
        #
        # # target network update op
        # self.update_target_op = []
        # t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)
        # e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
        # for i in range(len(t_params)):
        #     self.update_target_op.append(tf.assign(t_params[i], e_params[i]))
        #
        # # init tensorflow session
        # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        # self.sess.run(tf.global_variables_initializer())

        # init replay buffers
        self.replay_buf_len = 0
        self.memory_size = memory_size
        self.replay_buf_view = ReplayBuffer(shape=(memory_size,) + self.view_space)
        self.replay_buf_feature = ReplayBuffer(shape=(memory_size,) + self.feature_space)
        self.replay_buf_action = ReplayBuffer(shape=(memory_size,), dtype=np.int32)
        self.replay_buf_reward = ReplayBuffer(shape=(memory_size,))
        self.replay_buf_terminal = ReplayBuffer(shape=(memory_size,), dtype=np.bool)
        self.replay_buf_mask = ReplayBuffer(shape=(memory_size,))
        # if mask[i] == 0, then the item is used for padding, not for training

    def create_network(self, input_view, input_feature):
        kernel_num = [64, 64, 32]
        hidden_size = 256

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_view.size(0), kernel_num[0], kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(kernel_num[0], kernel_num[1], kernel_size=3)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(kernel_num[1], kernel_num[2], kernel_size=3)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_view(1))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_view(2))))
        linear_input_size = convw * convh * kernel_num[2] + input_feature.size(0)
        self.head = nn.Linear(linear_input_size, self.num_actions) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x_view, x_feature):
        x = F.relu(self.conv1(x_view))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(torch.cat((x.view(x.size(0), -1), x_feature)))


    def infer_action(self, raw_obs, ids, policy='e_greedy', eps=0):
        """infer action for a batch of agents

        Parameters
        ----------
        raw_obs: tuple(numpy array, numpy array)
            raw observation of agents tuple(views, features)
        ids: numpy array
            ids of agents
        policy: str
            can be eps-greedy or greedy
        eps: float
            used when policy is eps-greedy

        Returns
        -------
        acts: numpy array of int32
            actions for agents
        """
        self.eval()
        view, feature = raw_obs[0], raw_obs[1]

        if policy == 'e_greedy':
            eps = eps
        elif policy == 'greedy':
            eps = 0

        n = len(view)
        batch_size = min(n, self.infer_batch_size)

        # infer by splitting big batch in serial
        ret = []
        for i in range(0, n, batch_size):
            beg, end = i, i + batch_size
            ret.append(self.forward(view[beg:end], feature[beg:end]).argmax(1))
                # self.sess.run(self.output_action, feed_dict={
                # self.input_view: view[beg:end],
                # self.input_feature: feature[beg:end],
                # self.eps: eps}))
        ret = np.concatenate(ret)
        return ret







