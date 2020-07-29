import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
#sys.path.append(os.path.join(BASE_DIR, '../pointnet'))
#import tf_util
from transform_nets import input_transform_net
from utils import tf_util
import copy as cp


def get_model(point_cloud, is_training, n_classes=8, bn_decay=None):
 with tf.variable_scope('Seseg_'):
  """ Classification PointNet, input is BxNx3, output Bx40 """

  """ ConvNet baseline, input is BxNx9 gray image """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)

  k = 6
  bn = False
  weight_decay = .9

 

  batch_size = point_cloud.get_shape()[0].value
  batch_size_ = point_cloud.get_shape()[1].value
  n_features = 256
  patt = point_cloud[:, :, n_features+3:]  
  adj = tf_util.pairwise_distance(point_cloud[:, :, n_features:3 + n_features])

  nn_idx = tf_util.knn(adj, k=k)  

  pc_aux = point_cloud[:, :, :n_features]
  point_cloud = pc_aux

  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  out1 = tf_util.conv2d(edge_feature, 64, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=bn, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv1', bn_decay=bn_decay, is_dist=True)

  out2 = tf_util.conv2d(out1, 128, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=bn, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

  adj = tf_util.pairwise_distance(net_1)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

  out3 = tf_util.conv2d(edge_feature, 256, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=bn, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  out4 = tf_util.conv2d(out3, 512, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=bn, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv4', bn_decay=bn_decay, is_dist=True)

  net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

  adj = tf_util.pairwise_distance(net_2)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

  out5 = tf_util.conv2d(edge_feature, 256, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=bn, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

 
  net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)

  out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 256, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=bn, is_training=is_training,
                        scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand,
                                     net_1,
                                     net_2,
                                     net_3])

  # CONV
  net = tf_util.conv2d(concat, 128, [1, 1], padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training, scope='seg/conv1', is_dist=True)
  net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training, scope='seg/conv2', is_dist=True)
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
  net = tf_util.conv2d(net, 32, [1, 1], padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training, scope='seg/conv3', is_dist=True)
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
  net = tf_util.conv2d(net, 16, [1, 1], padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training, scope='seg/conv4', is_dist=True)
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp3')
  net = tf_util.conv2d(net, n_classes, [1, 1], padding='VALID', stride=[1, 1],
                       activation_fn=None, scope='seg/conv5', is_dist=True)
  net = tf.squeeze(net, [2])
  net = tf.squeeze(net, [0])

  return net
