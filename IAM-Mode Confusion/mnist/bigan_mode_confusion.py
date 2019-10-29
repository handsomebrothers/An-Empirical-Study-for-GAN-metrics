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

import torch.utils.data as Data
import os
if not os.path.isdir('saved_models_{}'.format('bigan')):
    os.mkdir('saved_models_{}'.format('bigan'))
f = open('saved_models_{}/log_collapse1.txt'.format('bigan'), mode='w')
import torch.utils.data as Data

from keras.datasets import mnist,fashion_mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import os
import matplotlib.pyplot as plt

import numpy as np

class BIGAN():
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
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)


    def build_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (X_test, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
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
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                z = np.random.normal(size=(batch_size, self.latent_dim))
                imgs_ = self.generator.predict(z)

                # Select a random batch of images and encode
                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                # imgs = X_train[idx]
                z_ = self.encoder.predict(image_batch)

                # Train the discriminator (img -> z is valid, z -> img is fake)
                d_loss_real = self.discriminator.train_on_batch([z_, image_batch], valid)
                d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (z -> img is valid and img -> z is is invalid)
                g_loss = self.bigan_generator.train_on_batch([z, image_batch], [valid, fake])

                # Plot the progress
                print("epoch:%d step:%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch,global_step, d_loss[0],
                                                                                   100 * d_loss[1], g_loss[0]))

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
    bigan = BIGAN()
    bigan.train(epochs=30, batch_size=64, sample_interval=200)
