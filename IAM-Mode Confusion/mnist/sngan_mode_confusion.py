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
if not os.path.isdir('saved_models_{}'.format('sngan')):
    os.mkdir('saved_models_{}'.format('sngan'))
f = open('saved_models_{}/log_collapse1.txt'.format('sngan'), mode='w')
import torch.utils.data as Data

import torch.utils.data as Data
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10,fashion_mnist,mnist
import os
from scipy import misc
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D,LeakyReLU,Conv2DTranspose, Conv2D
from keras.optimizers import Adam
import os
from keras.layers.convolutional import _Conv
from keras.legacy import interfaces
from keras.engine import InputSpec
import cv2
import keras.backend.tensorflow_backend as KTF

from scipy import misc
def set_gpu_config(device = "0",fraction=0.25):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.visible_device_list = device
    KTF.set_session(tf.Session(config=config))


def predict_images(file_name, generator, noise_size, n = 10, size = 32):

    image = generator.predict(np.random.normal(size=(n*n, ) + noise_size))

    image = np.reshape(image, (n, n, size, size, 3))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    image = np.reshape(image, (n*size, n*size, 3))

    image = 255 * (image + 1) / 2
    image = image.astype("uint8")
    misc.imsave(file_name, image)
def build_generator(input_shape):
    model = Sequential()

    model.add(Conv2DTranspose(512,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(128,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(64,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1,(3,3),padding="same",activation="tanh"))
    model.summary()
    return model


def build_discriminator(input_shape):
    model = Sequential()

    model.add(SNConv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(128,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(256,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(512,(3,3),strides=(2,2),padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(SNConv2D(1,(3,3),padding="same"))
    model.add(GlobalAveragePooling2D())
    model.summary()

    return model

def build_functions(batch_size, noise_size, image_size, generator, discriminator):

    noise = K.random_normal((batch_size,) + noise_size,0.0,1.0,"float32")
    real_image = K.placeholder((batch_size,) + image_size)
    fake_image = generator(noise)

    d_input = K.concatenate([real_image, fake_image], axis=0)
    pred_real, pred_fake = tf.split(discriminator(d_input), num_or_size_splits = 2, axis = 0)

    d_loss = K.mean(K.maximum(0., 1 - pred_real)) + K.mean(K.maximum(0., 1 + pred_fake))
    g_loss = -K.mean(pred_fake)

    d_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(d_loss, discriminator.trainable_weights)
    d_train = K.function([real_image, K.learning_phase()], [d_loss], d_training_updates)

    g_training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(g_loss, generator.trainable_weights)
    g_train = K.function([real_image, K.learning_phase()], [g_loss], g_training_updates)

    return d_train,g_train

class SNConv2D(_Conv):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.Ip = 1
        self.u = self.add_weight(
            name='W_u',
            shape=(1,filters),
            initializer='random_uniform',
            trainable=False
        )

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.W_bar(),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(SNConv2D, self).get_config()
        config.pop('rank')
        return config

    def W_bar(self):
        # Spectrally Normalized Weight
        W_mat = K.permute_dimensions(self.kernel, (3, 2, 0, 1)) # (h, w, i, o) => (o, i, h, w)
        W_mat = K.reshape(W_mat,[K.shape(W_mat)[0], -1]) # (o, i * h * w)

        if not self.Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = _l2normalize(K.dot(_u, W_mat))
            _u = _l2normalize(K.dot(_v, K.transpose(W_mat)))

        sigma = K.sum(K.dot(_u,W_mat)*_v)

        K.update(self.u,K.in_train_phase(_u, self.u))
        return self.kernel / sigma

def _l2normalize(x):
    return x / K.sqrt(K.sum(K.square(x)) + K.epsilon())
set_gpu_config("0",0.5)

epochs = 50
image_size = (32,32,1)
noise_size = (2,2,32)
batch_size = 64
sample_size=10
size=32
sample_interval=200
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=[cv2.resize(i,(32,32)) for i in x_train]

x_train=np.array(x_train)
x_test=np.array(x_test)
x_train=np.expand_dims(x_train,axis=3)
x_test=np.expand_dims(x_test,axis=3)
num_of_data = x_train.shape[0]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = (x_train/255)*2-1
x_test = (x_test/255)*2-1
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
x = []
y = np.zeros((31, 1), dtype=np.int)
y = list(y)
for i in range(31):
    y[i] = []
generator = build_generator(noise_size)
discriminator = build_discriminator(image_size)
d_train, g_train = build_functions(batch_size, noise_size, image_size, generator, discriminator)

nb_batches = int(x_train.shape[0] / batch_size)
global_step = 0
for epoch in range(epochs):
    for index in range(nb_batches):
        global_step += 1
        real_images = x_train[index * batch_size:(index + 1) * batch_size]
        d_loss, = d_train([real_images, 1])
        g_loss, = g_train([real_images, 1])
        print("[{0}/{1}] [{2}_{3}] d_loss: {4:.4}, g_loss: {5:.4}".format(epoch, epochs, epoch, global_step, d_loss,
                                                                              g_loss))
        sampleSize=5000
        if global_step % sample_interval == 0:
            image = generator.predict(np.random.normal(size=(sampleSize,) + noise_size))
            image = 255 * (image + 1) / 2

            image = [cv2.resize(i, (28,28)) for i in image]
            image = np.expand_dims(image, axis=3)
            image=np.array(image)
            y_logits = get_possibility(image)
            metrics = get_mean_var(y_logits)
            print(np.array(metrics).shape)
            f.writelines('\n')
            f.writelines('epoch:' + str(global_step))
            f.writelines('\n')
            f.writelines(' %.8f ' % (i) for i in metrics)
            f.writelines('\n')
