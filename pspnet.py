#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hanyin Fang <fhy881229@163.com>

import argparse
import os

import tensorflow as tf
from tensorpack import *
from tensorflow.contrib.layers import variance_scaling_initializer

from ThorDataFlow import CocoDataFlow

"""
Implementation of Pyramid Scene Parsing Network (PSPNet) based on Tensorpack
"""

BATCH_SIZE = 16
SHAPE = 28
CHANNELS = 3


class Model(ModelDesc):
  def _get_inputs(self):
    x = InputDesc(tf.float32, (None, SHAPE, SHAPE, CHANNELS), 'input')
    y = InputDesc(tf.float32, (None, SHAPE, SHAPE, CHANNESL), 'label')
    return [x, y]

  def _build_graph(self, inputs):
    image, label = inputs
    image = tf.cast(image, tf.float32) * (1.0 / 255)

    img_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    img_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image = (image - img_mean) / img_std
    if self.data_format == 'NCHW':
      image = tf.transpose(image, [0, 3, 1, 2])

    def shortcut(l, n_in, n_out, stride):
      if n_in != n_out:
        return Conv2D('convshortcut', l, n_out, 1, stride=stride)
      else:
        return l

    def basicblock(l, ch_out, stride, preact):
      ch_in = l.get_shape().as_list()[1]
      if preact == 'both_preact':
        l = BNReLU('preact', l)
        input = l
      elif preact == 'no_preact':
        input = l
        l = BNReLU('preact', l)
      else:
        input = l
      l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
      l = Conv2D('conv2', l, ch_out, 3)
      return l + shortcut(input, ch_in, ch_out, stride)

    def bottleneck(l, ch_out, stride, preact):
      ch_in = l.get_shape().as_list()[1]
      if preact == 'both_preact':
        l = BNReLU('preact', l)
        input = l
      elif preact == 'no_preact':
        input = l
        l = BNReLU('preact', l)
      else:
        input = l
      l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
      l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
      l = Conv2D('conv3', l, ch_out*4, 1)
      return l + shortcut(input, ch_in, ch_out*4, stride)

    def layer(l, layername, block_func, features, count, stride, first=False):
      with tf.variable_scope(layername):
        with tf.variable_scope('block0'):
          l = block_func(l, features, stride,
                         'no_preact' if first else 'both_preact')
        for i in range(1, count):
          with tf.variable_scope('block{}'.format(i)):
            l = block_func(l, features, 1, 'default')
        return l

    defs, block_func = ([3, 4, 6, 3], bottleneck)

    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                     W_init=variance_scaling_initializer(mode='FAN_OUT')),
         argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.dataformat):
      logits = (LinearWarp(image)
                .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                .apply(layer, 'group1', block_func, 128, defs[1], 2)
                .apply(layer, 'group2', block_func, 256, defs[2], 2)
                .apply(layer, 'group3', block_func, 512, defs[3], 2)())

    self.cost = tf.identity(0., name='total_costs')
    summary.add_moving_summary(self.cost)

  def _get_optimizer(self):
      lr = symbolic_functions.get_scalar_var(
          'learning_rate', 5e-3, summary=True)
      return tf.train.AdamOptimizer(lr)


def get_data(subset):
        # something that yields [[SHAPE, SHAPE, CHANNELS], [1]]
    ds = FakeData([[SHAPE, SHAPE, CHANNELS], [1]], 1000, random=False,
                  dtype=['float32', 'uint8'], domain=[(0, 255), (0, 10)])
    ds = PrefetchDataZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    logger.auto_set_dir()

    ds_train = get_data('train')
    ds_test = get_data('test')

    return TrainConfig(
        model=Model(),
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_test, [ScalarStats('total_costs')]),
        ],
        steps_per_epoch=ds_train.size(),
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    config.nr_tower = get_nr_gpu()

    if args.load:
        config.session_init = SaverRestore(args.load)

    SyncMultiGPUTrainer(config).train()
