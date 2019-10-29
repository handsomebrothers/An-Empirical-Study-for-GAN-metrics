#-*-coding:utf-8-*-

import tensorflow as tf
import os
from keras.models import Sequential, load_model
model = load_model('./models/model.h5')
model.load_weights('./models/weights.best.hdf5')
def get_mean_var(y_logit):
    y_logit=np.abs(y_logit-0.1)
    return np.mean(y_logit,axis=1)

def get_possibility(images):
    y_logits = model.predict(np.array(images))
    y_logits = tf.convert_to_tensor(y_logits)
    y_logits = tf.nn.softmax(y_logits)
    y_logits = tf.Session().run(y_logits)
    return y_logits
if not os.path.isdir('saved_models_{}'.format('bigan')):
    os.mkdir('saved_models_{}'.format('bigan'))
f = open('saved_models_{}/log_collapse1.txt'.format('bigan'), mode='w')
import cv2
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
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(self.latent_dim))

        model.summary()
        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):


        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(28*28*128)(d_in)
        model=Reshape((28,28,128))(model)
        model=Conv2D(32, kernel_size=3, strides=2,  padding="same")(model)
        model=LeakyReLU(alpha=0.2)(model)
        model=Dropout(0.25)(model)
        model=Conv2D(64, kernel_size=3, strides=2, padding="same")(model)
        model=ZeroPadding2D(padding=((0, 1), (0, 1)))(model)
        model=BatchNormalization(momentum=0.8)(model)
        model=LeakyReLU(alpha=0.2)(model)
        model=Dropout(0.25)(model)
        model=Conv2D(128, kernel_size=3, strides=2, padding="same")(model)
        model=BatchNormalization(momentum=0.8)(model)
        model=LeakyReLU(alpha=0.2)(model)
        model=Dropout(0.25)(model)
        model=Conv2D(256, kernel_size=3, strides=1, padding="same")(model)
        model=BatchNormalization(momentum=0.8)(model)
        model=LeakyReLU(alpha=0.2)(model)
        model=Dropout(0.25)(model)
        model=Flatten()(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (X_test, _) = fashion_mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        X_test = np.expand_dims(X_test, axis=3)
        y_train = y_train.reshape(-1, 1)
        # 
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        nb_batches = int(X_train.shape[0] / batch_size)
        global_step = 0

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            for index in range(nb_batches):
                global_step += 1
                # progress_bar.update(index)

                # get a batch of real images
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
                print("epoch:%d step:%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, global_step, d_loss[0],
                                                                                   100 * d_loss[1], g_loss[0]))

                sampleSize = 5000
                # If at save interval => save generated image samples
                if global_step % sample_interval == 0:
                    metrics = self.sample_images(global_step, sampleSize)

    def sample_images(self, epoch, sampleSize):
        r, c = sampleSize // 10, 10
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
    bigan.train(epochs=40, batch_size=64, sample_interval=200)
