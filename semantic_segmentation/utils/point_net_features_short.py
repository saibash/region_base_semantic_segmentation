import tensorflow as tf
import numpy as np
from pointnet import tf_util,transform_nets_short

from pointnet import tf_util



# # global parameter
# num_point = 1
# def set_num_point(value):
#     global num_point
#     num_point = value
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
DECAY_STEP = float(200000)
BN_DECAY_CLIP = 0.99
BASE_LEARNING_RATE=0.001
DECAY_RATE=.7

#from mayavi import mlab
import os

def get_learning_rate(batch,BATCH_SIZE):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch,batch_size):
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(200000)
    BN_DECAY_CLIP = 0.99
    BATCH_SIZE=batch_size

    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def model_v(point_cloud,is_training):
  with tf.device('/device:CPU:0'):
    with tf.variable_scope("modelv") as mv:
        #is_training = tf.constant(True)

        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value



        bn_decay=None

        with tf.variable_scope('transform_net1') as scope:
            transform = transform_nets_short.input_transform_net(point_cloud, is_training,bn_decay=bn_decay)

        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)



        net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)

        net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = transform_nets_short.feature_transform_net(net, is_training, bn_decay=bn_decay, K=64)

        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])


        net = tf_util.conv2d(net_transformed, 32, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)

        net = tf_util.conv2d(net, 16, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)

        net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool')

        net = tf.reshape(net, [batch_size, -1])

        return net



