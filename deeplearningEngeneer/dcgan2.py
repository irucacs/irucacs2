import tensorflow as tf
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display

from tensorflow.keras.layers import Dense, BatchNormalization, Conv2DTranspose, Reshape, LeakyReLU, Conv2D, Dropout, Flatten
from tensorflow.keras.activations import tanh

BATCH_SIZE = 1
noise_dim = 100

# 損失関数は本物と偽物の分類問題になりますのでバイナリクロスエントロピーを使用します
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    # fake imageに対してのloss算出
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    # 引数：real_outputは本物の画像, fake_outputは本物の画像
    # real imageに対してのloss算出
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # fake imageに対してのloss算出
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # totalのlossを算出
    total_loss = real_loss + fake_loss
    return total_loss


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = Dense(7*7*256, use_bias=False)
        self.batchnorm1 = BatchNormalization()
        self.relu1 = LeakyReLU()
        
        self.conv1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batchnorm2 = BatchNormalization()
        self.relu2 = LeakyReLU()

        self.conv2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = BatchNormalization()
        self.relu3 = LeakyReLU()

        self.conv3 = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = self.relu1(x)

        x = tf.reshape(x, shape=(-1, 7, 7, 256))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = self.relu2(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = self.relu3(x)

        x = self.conv3(x)
        x = tanh(x) 
        return x

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.relu1 = LeakyReLU()
        self.dropout1 = Dropout(0.3)

        self.conv2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.relu2 = LeakyReLU()
        self.dropout2 = Dropout(0.3)

        self.flatten = Flatten()
        self.fc1 = Dense(1)
    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)

        x = self.flatten(x)
        x = self.fc1(x)
        return x

# GeneratorとDiscriminatorのクラスを呼び出す
generator = Generator()
discriminator = Discriminator()

def train_step(images):
    # インプットはimage（本物画像）です
    # バッチサイズでノイズを作成
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    print(noise)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # ノイズを生成器に入力して画像を生成
        generated_images = generator(noise, training=True)
        # 画像を識別器に入力して真偽を出力
        # 本物画像
        real_output = discriminator(images, training=True)
        # Generatorで作成した偽物画像
        fake_output = discriminator(generated_images, training=True)
        # lossを算出
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    # 勾配を算出
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # 勾配をもとに最適化
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    


