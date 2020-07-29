import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))

from transform_nets_soft import input_transform_net, feature_transform_net
from utils import tf_util

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl


def get_model(point_cloud, labels, is_training, class_weights, n_classes=8, bn_decay=None):
 with tf.variable_scope('Features_', reuse=tf.AUTO_REUSE):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}

  bn = False
  bn_decay = 0.9
  
  ######################
  with tf.variable_scope('transform_net1') as sc:
      transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
  end_points['transform'] = transform
  point_cloud_transformed = tf.matmul(point_cloud, transform)

  input_image = tf.expand_dims(point_cloud_transformed, -1)

  net = tf_util.conv2d(input_image, 64, [1, 3],
                       padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training,
                       scope='conv1', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training,
                       scope='conv2', bn_decay=bn_decay)
 
  net = tf_util.conv2d(net, 256, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training,
                       scope='conv3', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 256, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training,
                       scope='conv4', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 256, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=bn, is_training=is_training,
                       scope='conv5', bn_decay=bn_decay) # 512

  # Symmetric function: max pooling
  net = tf_util.max_pool2d(net, [num_point, 1],
                           padding='VALID', scope='maxpool')

  global_features = tf.reshape(net, [batch_size, -1])

  net = tf_util.fully_connected(global_features, 128, bn=bn, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay) 
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                        scope='dp1')
  net = tf_util.fully_connected(net, 64, bn=bn, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                        scope='dp2')
  net = tf_util.fully_connected(net, n_classes, activation_fn=None, scope='fc3')

  classify_loss = get_loss(net, label=labels, class_weights=class_weights, n_classes=n_classes)


  return global_features, classify_loss, net

def get_loss(pred, label, class_weights, n_classes):
     """ pred: B*NUM_CLASSES,
         label: B, """
     
     with tf.variable_scope('loss') as scope:
 
         # HANDLE CLASS WEIGHTS

         class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
         label_weights = tf.gather(class_weights_tensor, indices=tf.reshape(label, (-1,)))
        
         # CACLULATE LOSSES
         classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(label, (-1,)), logits=tf.reshape(pred, (-1, n_classes)), weights=label_weights,
                                                reduction="weighted_sum_by_nonzero_weights")

         # SUMS ALL LOSSES - even Regularization losses automatically
         #classify_loss = tf.losses.get_total_loss()

     return classify_loss




