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
if not os.path.isdir('saved_models_{}'.format('lsgan')):
    os.mkdir('saved_models_{}'.format('lsgan'))
f = open('saved_models_{}/log_collapse1.txt'.format('lsgan'), mode='w')
import os
import tensorflow as tf


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

class LSGAN():
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
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(z, valid)
        # (!!!) Optimize w.r.t. MSE loss instead of crossentropy
        self.combined.compile(loss='mse', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # (!!!) No softmax
        model.add(Dense(1))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (X_test, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        # Rescale -1 to 1
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        # X_test = X_test / 127.5 - 1.
        X_test = np.expand_dims(X_test, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        nb_batches = int(X_train.shape[0] / batch_size)
        global_step = 0

        for epoch in range(epochs):

            for index in range(nb_batches):
                global_step += 1
                imgs = X_train[index * batch_size:(index + 1) * batch_size]
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print("epoch:%d step:%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, global_step,d_loss[0], 100 * d_loss[1], g_loss))

                sampleSize = 5000
                # If at save interval => save generated image samples
                if global_step % sample_interval == 0:
                    metrics = self.sample_images(global_step, sampleSize)

    def sample_images(self, epoch, sampleSize):
        r, c = 10, sampleSize // 10
        noise = np.random.normal(0, 1, (r * c, 100))
        # sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
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
    gan = LSGAN()
    gan.train(epochs=30, batch_size=64, sample_interval=200)
