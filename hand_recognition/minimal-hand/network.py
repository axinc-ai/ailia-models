import os

import numpy as np
import tensorflow as tf

################################# detection ####################################

def zero_padding(inputs, pad_1, pad_2):
  pad_mat = np.array([[0, 0], [pad_1, pad_2], [pad_1, pad_2], [0, 0]])
  return tf.pad(inputs, paddings=pad_mat)


def conv_bn(inputs, oc, ks, st, scope, training, rate=1):
  with tf.variable_scope(scope):
    if st == 1:
      layer = tf.layers.conv2d(
        inputs, oc, ks, strides=st, padding='SAME', use_bias=False,
        dilation_rate=rate,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
        kernel_initializer=tf.contrib.layers.xavier_initializer()
      )
    else:
      pad_total = ks - 1
      pad_1 = pad_total // 2
      pad_2 = pad_total - pad_1
      padded_inputs = zero_padding(inputs, pad_1, pad_2)
      layer = tf.layers.conv2d(
        padded_inputs, oc, ks, strides=st, padding='VALID', use_bias=False,
        dilation_rate=rate,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
        kernel_initializer=tf.contrib.layers.xavier_initializer()
      )
    layer = tf.layers.batch_normalization(layer, training=training)
  return layer


def conv_bn_relu(inputs, oc, ks, st, scope, training, rate=1):
  layer = conv_bn(inputs, oc, ks, st, scope, training, rate=rate)
  layer = tf.nn.relu(layer)
  return layer


def bottleneck(inputs, oc, st, scope, training, rate=1):
  with tf.variable_scope(scope):
    ic = inputs.get_shape().as_list()[-1]
    if ic == oc:
      if st == 1:
        shortcut = inputs
      else:
        shortcut = \
          tf.nn.max_pool2d(inputs, [1, st, st, 1], [1, st, st, 1], 'SAME')
    else:
      shortcut = conv_bn(inputs, oc, 1, st, 'shortcut', training)

    residual = conv_bn_relu(inputs, oc//4, 1, 1, 'conv1', training)
    residual = conv_bn_relu(residual, oc//4, 3, st, 'conv2', training, rate)
    residual = conv_bn(residual, oc, 1, 1, 'conv3', training)
    output = tf.nn.relu(shortcut + residual)

  return output


def resnet50(inputs, scope, training):
  with tf.variable_scope(scope):
    layer = conv_bn_relu(inputs, 64, 7, 2, 'conv1', training)

    with tf.variable_scope('block1'):
      for unit in range(2):
        layer = bottleneck(layer, 256, 1, 'unit%d' % (unit+1), training)
      layer = bottleneck(layer, 256, 2, 'unit3', training)

    with tf.variable_scope('block2'):
      for unit in range(4):
        layer = bottleneck(layer, 512, 1, 'unit%d' % (unit+1), training, 2)

    with tf.variable_scope('block3'):
      for unit in range(6):
        layer = bottleneck(layer, 1024, 1, 'unit%d' % (unit+1), training, 4)

    layer = conv_bn_relu(layer, 256, 3, 1, 'squeeze', training)

  return layer


def net_2d(features, training, scope, n_out):
  with tf.variable_scope(scope):
    layer = conv_bn_relu(features, 256, 3, 1, 'project', training)
    with tf.variable_scope('prediction'):
      hmap = tf.layers.conv2d(
        layer, n_out, 1, strides=1, padding='SAME',
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.01)
      )
  return hmap


def net_3d(features, training, scope, n_out, need_norm):
  with tf.variable_scope(scope):
    layer = conv_bn_relu(features, 256, 3, 1, 'project', training)
    with tf.variable_scope('prediction'):
      dmap_raw = tf.layers.conv2d(
        layer, n_out * 3, 1, strides=1, padding='SAME',
        activation=None,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.01)
      )
      if need_norm:
        dmap_norm = tf.norm(dmap_raw, axis=-1, keepdims=True)
        dmap = dmap_raw / tf.maximum(dmap_norm, 1e-6)
      else:
        dmap = dmap_raw

  h, w = features.get_shape().as_list()[1:3]
  dmap = tf.reshape(dmap, [-1, h, w, n_out, 3])

  if need_norm:
    return dmap, dmap_norm

  return dmap


def tf_hmap_to_uv(hmap):
  hmap_flat = tf.reshape(hmap, (tf.shape(hmap)[0], -1, tf.shape(hmap)[3]))
  argmax = tf.argmax(hmap_flat, axis=1, output_type=tf.int32)
  argmax_x = argmax // tf.shape(hmap)[2]
  argmax_y = argmax % tf.shape(hmap)[2]
  uv = tf.stack((argmax_x, argmax_y), axis=1)
  uv = tf.transpose(uv, [0, 2, 1])
  return uv


def get_pose_tile(N):
  pos_tile = tf.tile(
    tf.constant(
      np.expand_dims(
        np.stack(
          [
            np.tile(np.linspace(-1, 1, 32).reshape([1, 32]), [32, 1]),
            np.tile(np.linspace(-1, 1, 32).reshape([32, 1]), [1, 32])
          ], -1
        ), 0
      ), dtype=tf.float32
    ), [N, 1, 1, 1]
  )
  return pos_tile


def detnet(img, n_stack, training):
  features = resnet50(img, 'resnet', training)
  pos_tile = get_pose_tile(tf.shape(img)[0])
  features = tf.concat([features, pos_tile], -1)

  hmaps = []
  dmaps = []
  lmaps = []
  for i in range(n_stack):
    hmap = net_2d(features, training, 'hmap_%d' % i, 21)
    features = tf.concat([features, hmap], axis=-1)
    hmaps.append(hmap)

    dmap = net_3d(features, training, 'dmap_%d' % i, 21, False)
    features = tf.concat([features, tf.reshape(dmap, [-1, 32, 32, 21 * 3])], -1)
    dmaps.append(dmap)

    lmap = net_3d(features, training, 'lmap_%d' % i, 21, False)
    features = tf.concat([features, tf.reshape(lmap, [-1, 32, 32, 21 * 3])], -1)
    lmaps.append(lmap)

  return hmaps, dmaps, lmaps


##################################### IK #######################################


def dense(layer, n_units):
  layer = tf.layers.dense(
    layer, n_units, activation=None,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
    kernel_initializer=tf.initializers.truncated_normal(stddev=0.01)
  )
  return layer


def dense_bn(layer, n_units, training):
  layer = dense(layer, n_units)
  layer = tf.layers.batch_normalization(layer, training=training)
  return layer


def iknet(xyz, depth, width, training):
  N = xyz.get_shape().as_list()[0]
  layer = tf.reshape(xyz, [N, -1])
  for _ in range(depth):
    layer = dense_bn(layer, width, training)
    layer = tf.nn.sigmoid(layer)
  theta_raw = dense(layer, 21 * 4)
  theta_raw = tf.reshape(theta_raw, [-1, 21, 4])
  eps = np.finfo(np.float32).eps
  norm = tf.maximum(tf.norm(theta_raw, axis=-1, keepdims=True), eps)
  theta_pos = theta_raw / norm
  theta_neg = theta_pos * -1
  theta = tf.where(
    tf.tile(theta_pos[:, :, 0:1] > 0, [1, 1, 4]), theta_pos, theta_neg
  )
  return theta, norm
