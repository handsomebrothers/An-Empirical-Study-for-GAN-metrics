#-*-coding:utf-8-*-
import util
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import os
MNIST_CLASSIFIER_FROZEN_GRAPH = '/home/ubuntu/soft/huangdengrong/GAN/Metrics-GAN/dataset/classify_mnist_graph_def.pb'
INPUT_TENSOR = 'inputs:0'
OUTPUT_TENSOR = 'logits:0'
def get_mean_var(y_logit):
    y_logit=np.abs(y_logit-0.1)
    return np.mean(y_logit,axis=1)

def get_possibility(images):
    eval_images = tf.convert_to_tensor(images)
    y_logits = util.mnist_logits(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH, INPUT_TENSOR, OUTPUT_TENSOR)
    y_logits = tf.Session().run(tf.nn.softmax(y_logits))
    # print(y_logits)
    return y_logits
if not os.path.isdir('saved_models_{}'.format('cogan')):
    os.mkdir('saved_models_{}'.format('cogan'))
f = open('saved_models_{}/log_collapse1.txt'.format('cogan'), mode='w')

import os
import tensorflow as tf
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import os
import numpy as np


class COGAN():
    """Reference: https://wiseodd.github.io/techblog/2017/02/18/coupled_gan/"""

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.x = []
        self.y = np.zeros((31, 1), dtype=np.int)
        self.y = list(self.y)
        for i in range(31):
            self.y[i] = []

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.d1, self.d2 = self.build_discriminators()
        self.d1.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        self.d2.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])

        # Build the generator
        self.g1, self.g2 = self.build_generators()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img1 = self.g1(z)
        img2 = self.g2(z)

        # For the combined model we will only train the generators
        self.d1.trainable = False
        self.d2.trainable = False

        # The valid takes generated images as input and determines validity
        valid1 = self.d1(img1)
        valid2 = self.d2(img2)

        # The combined model  (stacked generators and discriminators)
        # Trains generators to fool discriminators
        self.combined = Model(z, [valid1, valid2])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                              optimizer=optimizer)

    def build_generators(self):

        # Shared weights between generators
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        noise = Input(shape=(self.latent_dim,))
        feature_repr = model(noise)

        # Generator 1
        g1 = Dense(1024)(feature_repr)
        g1 = LeakyReLU(alpha=0.2)(g1)
        g1 = BatchNormalization(momentum=0.8)(g1)
        g1 = Dense(np.prod(self.img_shape), activation='tanh')(g1)
        img1 = Reshape(self.img_shape)(g1)

        # Generator 2
        g2 = Dense(1024)(feature_repr)
        g2 = LeakyReLU(alpha=0.2)(g2)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Dense(np.prod(self.img_shape), activation='tanh')(g2)
        img2 = Reshape(self.img_shape)(g2)

        model.summary()

        return Model(noise, img1), Model(noise, img2)

    def build_discriminators(self):

        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)

        # Shared discriminator layers
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        img1_embedding = model(img1)
        img2_embedding = model(img2)

        # Discriminator 1
        validity1 = Dense(1, activation='sigmoid')(img1_embedding)
        # Discriminator 2
        validity2 = Dense(1, activation='sigmoid')(img2_embedding)

        return Model(img1, validity1), Model(img2, validity2)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (X_test, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        X_test = np.expand_dims(X_test, axis=3)

        # Images in domain A and B (rotated)
        X1 = X_train[:int(X_train.shape[0] / 2)]
        X2 = X_train[int(X_train.shape[0] / 2):]
        X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Select a random batch of images
            idx = np.random.randint(0, X1.shape[0], batch_size)
            imgs1 = X1[idx]
            imgs2 = X2[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images
            gen_imgs1 = self.g1.predict(noise)
            gen_imgs2 = self.g2.predict(noise)

            # Train the discriminators
            d1_loss_real = self.d1.train_on_batch(imgs1, valid)
            d2_loss_real = self.d2.train_on_batch(imgs2, valid)
            d1_loss_fake = self.d1.train_on_batch(gen_imgs1, fake)
            d2_loss_fake = self.d2.train_on_batch(gen_imgs2, fake)
            d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
            d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

            # ------------------
            #  Train Generators
            # ------------------

            g_loss = self.combined.train_on_batch(noise, [valid, valid])

            # Plot the progress
            print("%d [D1 loss: %f, acc.: %.2f%%] [D2 loss: %f, acc.: %.2f%%] [G loss: %f]" \
                  % (epoch, d1_loss[0], 100 * d1_loss[1], d2_loss[0], 100 * d2_loss[1], g_loss[0]))

            sampleSize = 5000
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                metrics = self.sample_images(epoch, sampleSize)

    def sample_images(self, epoch, sampleSize):
        r, c = 10, sampleSize // 10
        noise = np.random.normal(0, 1, (r * int(c / 2), 100))
        gen_imgs1 = self.g1.predict(noise)
        gen_imgs2 = self.g2.predict(noise)
        gen_imgs = np.concatenate([gen_imgs1, gen_imgs2])
        y_logits = get_possibility(gen_imgs)
        metrics = get_mean_var(y_logits)
        print(np.array(metrics).shape)
        f.writelines('\n')
        f.writelines('epoch:' + str(epoch))
        f.writelines('\n')
        f.writelines(' %.8f ' % (i) for i in metrics)
        f.writelines('\n')
        return metrics


if __name__ == '__main__':
    gan = COGAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)
#-*-coding:utf-8-*-