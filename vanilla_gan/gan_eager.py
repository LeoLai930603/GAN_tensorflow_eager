#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 上午10:23
# @Author  : Zhizhou Li
# @File    : gan_eager.py

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config

# Enable Eager Mode
tf.enable_eager_execution()

# The image size: 28 x 28 = 784
INPUT_SIZE = 784
# The input size of noise
NOISE_SIZE = 100
# Mini-batch size
BATCH_SIZE = 128


# Custom xavier initialization, which is not the same as xavier_init in tf
def xavier_init(size):
    '''
    :param size: tuple. For weight matrix initialization
    :return: tf.Tensor. A random-initialized Rank-2 tensor(matrix)
    '''
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Sampling algorithm of noise input
def sample_Z(m, n):
    '''
    :param m: int. Batch size
    :param n: int. Noise input size
    :return: np.ndarray. A random-initialized numpy array with shape=(m, n)
    '''
    z = np.random.uniform(-1.0, 1.0, size=[m, n])
    return z.astype(np.float32)


# For result observation from generator
def plot(samples: np.ndarray):
    '''
    :param samples: np.ndarray. The output of generator, shape => (batch_size, INPUT_SIZE)
    :return: figure created by matplotlib
    '''
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")
    return fig


# Computation of generator loss
def generator_loss(logits: tf.Tensor):
    '''
    :param logits: tf.Tensor. The output of generator, shape => (batch_size, INPUT_SIZE)
    :return: tf.Tensor. Average loss of generator
    '''
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))


# Compuation of discriminator loss
def discriminator_loss(real_logits: tf.Tensor, fake_logits: tf.Tensor):
    '''
    The loss contains two parts: loss from real input and loss from fake input
    :param real_logits: output of discriminator when input is from real source (natural images)
    :param fake_logits: output of discriminator when input is from fake source (generator output)
    :return: tf.Tensor. Combination of loss_real and loss_fake
    '''
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    return D_loss_real + D_loss_fake


# MLP-based Generator
class Generator:
    def __init__(self, hidden_size: int = 128):
        self.G_W1 = tf.Variable(xavier_init([NOISE_SIZE, hidden_size]))  # shape => (NOISE_SIZE, hidden_size)
        self.G_b1 = tf.Variable(tf.zeros(shape=[hidden_size]))  # shape => (hidden_size,)
        self.G_W2 = tf.Variable(xavier_init([hidden_size, INPUT_SIZE]))  # shape => (hidden_size, INPUT_SIZE)
        self.G_b2 = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))  # shape => (INPUT_SIZE,)
        self.theta_G = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]  # Trainable variables

    def forward(self, input_tensor):
        G_h1 = tf.nn.relu(tf.matmul(input_tensor, self.G_W1) + self.G_b1)  # shape => (batch_size, hidden_size)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2  # shape => (batch_size, INPUT_SIZE)
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob


# MLP-based Discriminator
class Discriminator:
    def __init__(self, hidden_size: int = 128):
        self.D_W1 = tf.Variable(xavier_init([INPUT_SIZE, hidden_size]))  # shape => (INPUT_SIZE, hidden_size)
        self.D_b1 = tf.Variable(tf.zeros(shape=[hidden_size]))  # shape => (hidden_size,)
        self.D_W2 = tf.Variable(xavier_init([hidden_size, 1]))  # shape => (hidden_size, 1) --- binary classification
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))  # shape => (1,)
        self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]  # Trainable variables

    def forward(self, input_tensor):
        D_h1 = tf.nn.relu(tf.matmul(input_tensor, self.D_W1) + self.D_b1)  # shape => (batch_size, hidden_size)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2  # shape => (batch_size, 1)
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit


def main():
    # setup the generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # prepare the data set, here we use mnist for a short demo
    mnist = input_data.read_data_sets(os.path.join(config.ROOT_DIR, "MNIST_data"), one_hot=True)

    # setup the optimizers for two MLP netowrks
    D_solver = tf.train.AdamOptimizer()
    G_solver = tf.train.AdamOptimizer()

    out_path = os.path.join(config.OUTPUT_DIR, "vanilla_gan")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Training start!
    i = 0
    for it in range(1000000):

        if it % 1000 == 0:
            samples = generator.forward(sample_Z(16, NOISE_SIZE))
            fig = plot(samples.numpy())
            file_path = os.path.join(out_path, "{}.png".format(str(i).zfill(3)))
            plt.savefig(file_path, bbox_inches="tight")
            i += 1
            plt.close(fig)

        # read data from data_set
        images, _ = mnist.train.next_batch(BATCH_SIZE)
        images = images.astype(np.float32)

        # acquire noise samples
        sample_noise = sample_Z(BATCH_SIZE, NOISE_SIZE)

        # forward computing of discriminator
        with tf.GradientTape() as tape_d:
            g_sample = generator.forward(sample_noise)
            d_real, d_logit_real = discriminator.forward(images)
            d_fake, d_logit_fake = discriminator.forward(g_sample)
            d_loss = discriminator_loss(d_logit_real, d_logit_fake)

        # backward propagation of discriminator
        grads = tape_d.gradient(d_loss, discriminator.theta_D)
        D_solver.apply_gradients(zip(grads, discriminator.theta_D), global_step=tf.train.get_or_create_global_step())

        # forward computing of generator
        sample_noise = sample_Z(BATCH_SIZE, NOISE_SIZE)
        with tf.GradientTape() as tape_g:
            g_sample = generator.forward(sample_noise)
            d_fake, d_logit_fake = discriminator.forward(g_sample)
            g_loss = generator_loss(d_logit_fake)

        # backward propagation of generator
        grads = tape_g.gradient(g_loss, generator.theta_G)
        G_solver.apply_gradients(zip(grads, generator.theta_G), global_step=tf.train.get_or_create_global_step())

        # logs every 1000 iteration
        if it % 1000 == 0:
            print("Iter: {}".format(it))
            print("D loss: {:.4}".format(d_loss))
            print("G loss: {:.4}".format(g_loss))
            print()


if __name__ == '__main__':
    main()
