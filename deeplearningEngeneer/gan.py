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

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images[:32]
train_labels = train_labels[:32]

# mnistのロード
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 32
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

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

generator = Generator()
discriminator = Discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # real imageに対してのloss算出
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # fake imageに対してのloss算出
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # totalのlossを算出
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # fake imageに対してのloss算出
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './6592_tensorflow_workbook_data/4.1.1_training_checkpoints' 
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(checkpoint_dir+'/ckpt-4')

EPOCHS = 1
noise_dim = 100
num_examples_to_generate = 16

# テストノイズを作成
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def generate_and_save_images(model, epoch, test_input):
    
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

@tf.function
def train_step(images):
    # バッチサイズでノイズを作成
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # ノイズを生成器に入力して画像を生成
        generated_images = generator(noise, training=True)
        # 画像を識別器に入力して真偽を出力
        real_output = discriminator(images, training=True)
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

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
    for image_batch in dataset:
        train_step(image_batch)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    
train(train_dataset, EPOCHS)