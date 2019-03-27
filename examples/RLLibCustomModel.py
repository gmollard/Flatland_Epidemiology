import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model

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
        Dense = tf.layers.dense

        conv1 = Relu(BatchNormalization(self.conv2d(input_dict['obs'][0], 64, 'valid'), axis=3,
                                        training=input_dict['is_training']))

        conv2 = Relu(BatchNormalization(self.conv2d(conv1, 64, 'valid'), axis=3, training=input_dict['is_training']))

        conv3 = Relu(BatchNormalization(self.conv2d(conv2, 64, 'valid'), axis=3, training=input_dict['is_training']))

        conv4 = Relu(BatchNormalization(self.conv2d(conv3, 64, 'valid'), axis=3, training=input_dict['is_training']))
        conv4_flat = tf.reshape(conv4, [-1, 64 * (31-2*4)**2])
        conv4_feature = tf.concat((conv4_flat, input_dict['obs'][1]), axis=1)
        s_fc1 = Dropout(
            Relu(BatchNormalization(Dense(conv4_feature, 1024, use_bias=False), axis=1, training=input_dict['is_training'])),
                rate=0.10)
        layerN_minus_1 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512, use_bias=False), axis=1, training=input_dict['is_training'])),
                        rate=0.1)
        layerN = Dense(layerN_minus_1, num_outputs)
        return layerN, layerN_minus_1

    def conv2d(self, x, out_channels, padding):
        return tf.layers.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding, use_bias=False)

