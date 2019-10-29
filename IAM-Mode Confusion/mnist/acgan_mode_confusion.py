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
if not os.path.isdir('saved_models_{}'.format('acgan')):
    os.mkdir('saved_models_{}'.format('acgan'))
f = open('saved_models_{}/log_collapse1.txt'.format('acgan'), mode='w')
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.utils.data as Data
class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100
        self.x = []
        self.y = np.zeros((31, 1), dtype=np.int)
        self.y = list(self.y)
        for i in range(31):
            self.y[i] = []

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (X_test, _) = mnist.load_data()

        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)
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
                # Image labels. 0-9
                img_labels = y_train[index * batch_size:(index + 1) * batch_size]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # The labels of the digits that the generator tries to create an
                # image representation of
                sampled_labels = np.random.randint(0, 10, (batch_size, 1))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, sampled_labels])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(image_batch, [valid, img_labels])
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

                # Plot the progress
                print("epoch:%d step:%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                    epoch, global_step, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

                sampleSize = 5000
                # If at save interval => save generated image samples
                if global_step % sample_interval == 0:
                    metrics=self.sample_images(global_step,sampleSize)
    def sample_images(self, epoch,sampleSize):
        r, c = 10, sampleSize//10
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        y_logits = get_possibility(gen_imgs)
        metrics = get_mean_var(y_logits)
        print(np.array(metrics).shape)
        f.writelines('\n')
        f.writelines('epoch:' + str(epoch))
        f.writelines('\n')
        f.writelines(' %.8f '%(i) for i in metrics)
        f.writelines('\n')
        return metrics



if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=30, batch_size=64, sample_interval=200)
