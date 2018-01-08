#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12.12.17 4:25 PM
# @Author  : Dieter_Lan

import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from tensorpack import *
import tensorflow.contrib.slim as slim
from bilinear_sampler import *

"""
This is a boiler-plate template.
All code is in this file is the most minimalistic way to solve a deep-learning problem with cross-validation.
"""

# BATCH_SIZE = 16
# SHAPE = 28
# HEIGHT = 1242
# WIDTH = 375
# CHANNELS = 3

BATCH_SIZE = 16
# SHAPE = 28
HEIGHT = 1280
WIDTH = 512
CHANNELS = 3


class ImageDecode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img
        super(ImageDecode, self).__init__(ds, func, index=index)


class Model(ModelDesc):

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        # input_img = img
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        # input_img = img
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def _get_inputs(self):   # this place ,I change NONE to BATCH_SIZE
        return [InputDesc(tf.float32, (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), 'left'),
                InputDesc(tf.float32, (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), 'right')]

    def _build_graph(self, inputs):
        # left is image [HEIGHT, WIDTH, 3] with range [0, 255]
        left, right = inputs
        # # left is image [HEIGHT, WIDTH, 3] with range [-1, 1]
        # left = left / 128 - 1
        # right = right / 128 - 1

        # START HERE
        # put the network

        def vgg_func(y):

            # with tf.variable_scope('encoder'):
            conv1 = self.conv_block(y, 32, 7)  # H/2
            conv2 = self.conv_block(conv1, 64, 5)  # H/4
            conv3 = self.conv_block(conv2, 128, 3)  # H/8
            conv4 = self.conv_block(conv3, 256, 3)  # H/16
            conv5 = self.conv_block(conv4, 512, 3)  # H/32
            conv6 = self.conv_block(conv5, 512, 3)  # H/64
            conv7 = self.conv_block(conv6, 512, 3)  # H/128

            # with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6


            upconv = self.upconv
            conv = self.conv
            # with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

            return [self.disp1, self.disp2, self.disp3, self.disp4]
#
        def some_func(y):
            x = Conv2D('conv1', y, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv2', x, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv3', x, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv4', x, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv5', x, 32, kernel_shape=3, nl=tf.nn.relu)
            x = Conv2D('conv6', x, 3, kernel_shape=3, nl=tf.identity) + y
            return x

        # STORE DISPARITIES
        disp_est = vgg_func(left)

        disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in disp_est]
        disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in disp_est]

        # # GENERATE IMAGES
        right_pyramid = self.scale_pyramid(right, 4)
        left_pyramid = self.scale_pyramid(left, 4)
        left_est  = [self.generate_image_left(right_pyramid[i], disp_left_est[i])  for i in range(4)]
        right_est = [self.generate_image_right(left_pyramid[i], disp_right_est[i]) for i in range(4)]



        # # LR CONSISTENCY
        right_to_left_disp = [self.generate_image_left(disp_right_est[i], disp_left_est[i]) for i in
                                    range(4)]
        left_to_right_disp = [self.generate_image_right(disp_left_est[i], disp_right_est[i]) for i in
                                    range(4)]


        # # DISPARITY SMOOTHNESS

        disp_left_smoothness = self.get_disparity_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.get_disparity_smoothness(disp_right_est, right_pyramid)



        # L1
        l1_left = [tf.abs(left_est[i] - left_pyramid[i]) for i in range(4)]
        l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in l1_left]
        l1_right = [tf.abs(right_est[i] - right_pyramid[i]) for i in range(4)]
        l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in l1_right]

        # SSIM
        ssim_left = [self.SSIM(left_est[i], left_pyramid[i]) for i in range(4)]
        ssim_loss_left = [tf.reduce_mean(s) for s in ssim_left]
        ssim_right = [self.SSIM(right_est[i], right_pyramid[i]) for i in range(4)]
        ssim_loss_right = [tf.reduce_mean(s) for s in ssim_right]


        # WEIGTHED SUM
        alpha_image_loss = 0.85
        image_loss_right = [
            alpha_image_loss * ssim_loss_right[i] + (1 - alpha_image_loss) *
            l1_reconstruction_loss_right[i] for i in range(4)]
        image_loss_left = [
            alpha_image_loss * ssim_loss_left[i] + (1 - alpha_image_loss) *
            l1_reconstruction_loss_left[i] for i in range(4)]
        image_loss = tf.add_n(image_loss_left + image_loss_right)

        # DISPARITY SMOOTHNESS
        disp_left_loss = [tf.reduce_mean(tf.abs(disp_left_smoothness[i])) / 2 ** i for i in range(4)]
        disp_right_loss = [tf.reduce_mean(tf.abs(disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        disp_gradient_loss = tf.add_n(disp_left_loss + disp_right_loss)

        # LR CONSISTENCY
        lr_left_loss = [tf.reduce_mean(tf.abs(right_to_left_disp[i] - disp_left_est[i])) for i in
                             range(4)]
        lr_right_loss = [tf.reduce_mean(tf.abs(left_to_right_disp[i] - disp_right_est[i])) for i in
                              range(4)]
        lr_loss = tf.add_n(lr_left_loss + lr_right_loss)

        # TOTAL LOSS
        disp_gradient_loss_weight =0.1  #default is 0.1
        lr_loss_weight = 1.0
        self.total_loss = image_loss + disp_gradient_loss_weight * disp_gradient_loss + lr_loss_weight * lr_loss

        # warped_left = some_func(left)
        #
        # corrected_left = 128. * (left + 1)
        # corrected_warped_left = 128. * (warped_left + 1)
        # corrected_right = 128. * (right + 1)

        # END HERE

        # viz = tf.concat([corrected_left, corrected_warped_left, corrected_right], 2)
        # viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        # tf.summary.image('left,pred,groundtruth', viz, max_outputs=max(30, BATCH_SIZE))

        # self.cost = tf.reduce_mean(tf.squared_difference(warped_left, right), name='total_costs')
        self.cost = self.total_loss
        summary.add_moving_summary(self.cost)

        # summary.add_moving_summary(self.total_loss)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-3, trainable=False)
        return tf.train.AdamOptimizer(lr)



def get_data():
    # ds = LMDBDataPoint('train2.lmdb', shuffle=True)
    ds = LMDBDataPoint('/graphics/projects/scratch/student_datasets/cgpraktikum17/DepthEstimation/KITTI/train.lmdb',
                       shuffle=True)
    ds = ImageDecode(ds, index=0)
    ds = ImageDecode(ds, index=1)

    def rescale(img):
        return cv2.resize(img, (WIDTH, HEIGHT))

    ds = MapDataComponent(ds, rescale, index=0)
    ds = MapDataComponent(ds, rescale, index=1)

    ds = PrefetchDataZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    # [(HEIGHT, WIDTH, 3), (HEIGHT, WIDTH, 3)]
    return ds


def get_config():
    logger.auto_set_dir()

    ds_train = get_data()

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver()
        ],
        steps_per_epoch=ds_train.size(),
        max_epoch=5,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()

    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    if args.load:
        config.session_init = SaverRestore(args.load)

    launch_train_with_config(config, SimpleTrainer())


    #
    #