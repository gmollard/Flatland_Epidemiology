import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import normc_initializer

import tensorflow as tf

class RLLibCustomModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].

        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:

        Examples:
            >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': (observation, features)
        """
        # print(input_dict)
        # Convolutional Layer #1

        Relu = tf.nn.relu
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.contrib.layers.fully_connected

        conv1 = Relu(self.conv2d(input_dict['obs'][0], 64, 'valid'))

        # conv2 = Relu(self.conv2d(conv1, 64, 'valid'))

        # conv3 = Relu(self.conv2d(conv2, 64, 'valid'))

        conv4_flat = tf.reshape(conv1, [-1, 64 * (31-2*1)**2])
        conv4_feature = tf.concat((conv4_flat, input_dict['obs'][1]), axis=1)
        s_fc1 = Relu(Dense(conv4_feature, 1024, use_bias=False))
        layerN_minus_1 = Relu(Dense(s_fc1, 512, use_bias=False))
        layerN = Dense(layerN_minus_1, num_outputs)
        return layerN, layerN_minus_1

    def conv2d(self, x, out_channels, padding):
        return tf.layers.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding, use_bias=False)


class LightModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].

        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:

        Examples:
            >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': (observation, features)
        """
        # print(input_dict)
        # Convolutional Layer #1
        self.sess = tf.get_default_session()
        Relu = tf.nn.relu
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.contrib.layers.fully_connected

        #conv1 = Relu(self.conv2d(input_dict['obs'][0], 32, 'valid'))
        conv1 = Relu(self.conv2d(input_dict['obs'], 32, 'valid'))
        conv2 = Relu(self.conv2d(conv1, 16, 'valid'))

        # conv3 = Relu(self.conv2d(conv2, 64, 'valid'))

        conv4_flat = tf.reshape(conv2, [-1, 16 * (17-2*2)**2])
        #conv4_feature = tf.concat((conv4_flat, input_dict['obs'][1]), axis=1)
        s_fc1 = Relu(Dense(conv4_flat, 128, weights_initializer=normc_initializer(1.0)))
        # layerN_minus_1 = Relu(Dense(s_fc1, 256, use_bias=False))
        layerN = Dense(s_fc1, num_outputs, weights_initializer=normc_initializer(0.01))
        return layerN, s_fc1

    def conv2d(self, x, out_channels, padding):
        return tf.layers.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding, use_bias=True)
                                # weights_initializer=normc_initializer(1.0))

    # def custom_loss(self, policy_loss, loss_inputs):
    #     """Override to customize the loss function used to optimize this model.
    #
    #     This can be used to incorporate self-supervised losses (by defining
    #     a loss over existing input and output tensors of this model), and
    #     supervised losses (by defining losses over a variable-sharing copy of
    #     this model's layers).
    #
    #     You can find an runnable example in examples/custom_loss.py.
    #
    #     Arguments:
    #         policy_loss (Tensor): scalar policy loss from the policy graph.
    #         loss_inputs (dict): map of input placeholders for rollout data.
    #
    #     Returns:
    #         Scalar tensor for the customized loss for this model.
    #     """
    #     # print("Los Inputs Shape: ", loss_inputs['obs'].shape)
    #     # print(policy_loss)
    #     # print_action_prob = tf.print(loss_inputs['actions'], summarize=200)
    #     # with tf.control_dependencies([print_action_prob]):
    #     #     tf.multiply(print_action_prob, print_action_prob)
    #     # self.sess.run()
    #     return tf.reduce_mean(policy_loss)

