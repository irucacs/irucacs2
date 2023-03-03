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

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # 全結合層
        self.fc1 = Dense(7*7*256, use_bias=False)
        self.batchnorm1 = BatchNormalization()
        self.relu1 = LeakyReLU()

        # アップサンプリング層1
        self.conv1 = Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batchnorm2 = BatchNormalization()
        self.relu2 = LeakyReLU()

        # アップサンプリング層2
        self.conv2 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = BatchNormalization()
        self.relu3 = LeakyReLU()

        # アップサンプリング層3
        self.conv3 = Conv2D(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    def call(self, x, training=True):
        # 全結合層
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = self.relu1(x)

        x = tf.reshape(x, shape=(-1, 7, 7, 256))

        # アップサンプリング層1
        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = self.relu2(x)

        # アップサンプリング層2
        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = self.relu3(x)

        # アップサンプリング層3
        x = self.conv3(x)
        
        # 出力
        x =self.fc(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 畳み込み層1
        self.conv1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.relu1 = LeakyReLU()
        self.dropout1 = Dropout(0.3)

        # 畳み込み層2
        self.conv2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.relu2 = LeakyReLU()
        self.dropout2 = Dropout(0.3)
        
        # 全結合層
        self.flatten = Flatten()
        self.fc1 = Dense(1)
    def call(self, x, training=True):
        # 畳み込み層1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)

        # 畳み込み層2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)

        # 全結合層
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# GeneratorとDiscriminatorのクラスを呼び出す
generator = Generator()
discriminator = Discriminator()