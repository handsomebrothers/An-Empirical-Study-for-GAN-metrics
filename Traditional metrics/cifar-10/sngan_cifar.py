#-*-coding:utf-8-*-
from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import Sequential
import util
import utils
import tensorflow.contrib.gan as tfgan
num_images_to_eval = 500
import torch.nn as nn
import tensorflow as tf
from PIL import Image

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
import numpy as np
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, imgs, transform=None):

        self.imgs = imgs
        self.transform = transform
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = img.transpose(2,0, 1)
        if self.transform is not None:

            img=transforms.ToTensor()(img)
        else:
            img = torch.from_numpy(img)


        return img


import math
import os
import numpy as np
import ot
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models

from scipy import linalg

from keras.datasets import mnist
import matplotlib.pyplot as plt

def giveName(iter):  # 7 digit name.
    ans = str(iter)
    return ans.zfill(7)

def make_dataset(dataset, dataroot, imageSize):
    """
    :param dataset: must be in 'cifar10 | lsun | imagenet | folder | lfw | fake'
    :return: pytorch dataset for DataLoader to utilize
    """
    if dataset in ['imagenet', 'folder', 'lfw']:
        print(os.getcwd() + dataroot)  # 函数的作用是用于返回当前工作目录
        # folder dataset
        # dataset = dset.ImageFolder(root=dataroot,
        dataset = dset.ImageFolder(root=os.getcwd() + dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(imageSize),
                                       # transforms.CenterCrop(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset == 'lsun':
        dataset = dset.LSUN(db_path=dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(imageSize),
                                transforms.CenterCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif dataset == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif dataset == 'celeba':
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(138),
                                       transforms.Resize(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    else:
        raise Exception('--dataset must be in cifar10 | lsun | imagenet | folder | lfw | fake')
    assert dataset
    return dataset

MNIST_CLASSIFIER_FROZEN_GRAPH = './classify_mnist_graph_def.pb'
INPUT_TENSOR = 'inputs:0'
OUTPUT_TENSOR = 'logits:0'
# CONV_TENSOR = 'fc3/Relu:0'
CONV_TENSOR = 'fc4/BiasAdd:0'
class ConvNetFeatureSaver(object):
    def __init__(self, model='cnn', workers=4, batchSize=64):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        self.batch_size = batchSize
        self.workers = workers
        if self.model.find('tfgan') >= 0:
            print('tfgan')

        elif self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).cuda().eval()
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.cuda().eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).cuda().eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).cuda().eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).cuda().eval()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError

    def save(self, imgFolder, dataloader, save2disk=False):
        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []

        for img in dataloader:
            with torch.no_grad():

                input  = img.type(torch.FloatTensor).cuda() # 转Float
                if self.model == 'tfgan':
                    gen_imgs = np.array(img)
                    eval_images = tf.convert_to_tensor(gen_imgs)
                    flogit = util.mnist_logits(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH, INPUT_TENSOR, OUTPUT_TENSOR)
                    fconv = util.mnist_logits(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH, INPUT_TENSOR, CONV_TENSOR)
                    flogit,fconv=tf.Session().run([flogit,fconv])

                    flogit=torch.from_numpy(flogit)
                    fconv=torch.from_numpy(fconv)
                elif self.model == 'vgg' or self.model == 'vgg16':
                    print(self.vgg.features(input).shape)
                    fconv = self.vgg.features(input).view(input.size(0), -1)  # 相当于reshape
                    flogit = self.vgg.classifier(fconv)
                    # flogit = self.vgg.logitifier(fconv)
                elif self.model.find('resnet') >= 0:
                    fconv = self.resnet_feature(
                        input).mean(3).mean(2).squeeze()
                    flogit = self.resnet.fc(fconv)
                elif self.model == 'inception' or self.model == 'inception_v3':
                    fconv = self.inception_feature(
                        input).mean(3).mean(2).squeeze()
                    flogit = self.inception.fc(fconv)

                else:
                    raise NotImplementedError
                fsmax = F.softmax(flogit)
                '''
                总共有四个空间：1.feature_pixl 2.feature_conv 3.feature_logit 4.feature_smax
                '''
                feature_pixl.append(img)
                feature_conv.append(fconv.data.cpu())
                feature_logit.append(flogit.data.cpu())
                feature_smax.append(fsmax.data.cpu())

        feature_pixl = torch.cat(feature_pixl, 0).to('cpu')
        feature_conv = torch.cat(feature_conv, 0).to('cpu')
        feature_logit = torch.cat(feature_logit, 0).to('cpu')
        feature_smax = torch.cat(feature_smax, 0).to('cpu')


        return feature_pixl, feature_conv, feature_logit, feature_smax

    # return feature_pixl, feature_conv, feature_logit, feature_smax


def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1)
    X2 = (X * X).sum(1).resize_(nX, 1)
    Y = Y.view(nY, -1)
    Y2 = (Y * Y).sum(1).resize_(nY, 1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2.0 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    ft = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_t = s.tp / (s.tp + s.fn)
    s.acc_f = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd


def entropy_score(X, Y, epsilons):
    Mxy = distance(X, Y, False)
    scores = []
    for epsilon in epsilons:
        scores.append(ent(Mxy.t(), epsilon))

    return scores


def ent(M, epsilon):
    n0 = M.size(0)
    n1 = M.size(1)
    neighbors = M.lt(epsilon).float()
    sums = neighbors.sum(0).repeat(n0, 1)
    sums[sums.eq(0)] = 1
    neighbors = neighbors.div(sums)
    probs = neighbors.sum(1) / n1
    rem = 1 - probs.sum()
    if rem < 0:
        rem = 0
    probs = torch.cat((probs, rem * torch.ones(1)), 0)
    e = {}
    e['probs'] = probs
    probs = probs[probs.gt(0)]
    e['ent'] = -probs.mul(probs.log()).sum()

    return e


eps = 1e-20


def inception_score(X):
    kl = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score


def mode_score(X, Y):
    kl1 = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0) + eps).log() - (Y.mean(0) + eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
            np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


class Score:
    emd = 0
    mmd = 0
    knn = None


def compute_score(real, fake, k=1, sigma=1, sqrt=True):
    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    s = Score()
    s.emd = wasserstein(Mxy, sqrt)
    s.mmd = mmd(Mxx, Mxy, Myy, sigma)
    s.knn = knn(Mxx, Mxy, Myy, k, sqrt)

    return s


'''
参数说明：
dataset:真实数据集的path
imageSize:图片的大小
dataroot_real:真实数据所在的path
batchSize
saveFolder_r:真实数据的保存位置
conv_model:卷积模型
'''


def compute_score_raw(real_dataloader, fake_dataloader, batchSize, saveFolder_r, saveFolder_f, conv_model='resnet34',
                      workers=4):
    convnet_feature_saver = ConvNetFeatureSaver(model=conv_model,
                                                batchSize=batchSize, workers=workers)
    print(saveFolder_r)
    print(saveFolder_f)
    feature_r = convnet_feature_saver.save(saveFolder_r, real_dataloader, False)
    feature_f = convnet_feature_saver.save(saveFolder_f, fake_dataloader, False)


    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(2 * 7 + 3)
    for i in range(0, 2):
        print('compute score in space: ' + str(i))
        Mxx = distance(feature_r[i], feature_r[i], False)
        Mxy = distance(feature_r[i], feature_f[i], False)
        Myy = distance(feature_f[i], feature_f[i], False)

        score[i * 7] = wasserstein(Mxy, True)
        score[i * 7 + 1] = mmd(Mxx, Mxy, Myy, 1)
        tmp = knn(Mxx, Mxy, Myy, 1, False)
        score[(i * 7 + 2):(i * 7 + 7)] = \
            tmp.acc, tmp.acc_t, tmp.acc_f, tmp.precision, tmp.recall


    score[14] = inception_score(feature_f[3])
    score[15] = mode_score(feature_r[3], feature_f[3])
    score[16] = fid(feature_r[3], feature_f[3])

    return score
labels_name=['w_pixl','mmd_pixl','acc_pixl','acc_t_pixl','acc_f_pixl','acc_precision_pixl','acc_recall_pixl',
             'w_conv','mmd_conv','acc_conv','acc_t_conv','acc_f_conv','acc_precision_conv','acc_recall_conv',
             'is','mode_score','fid']
if not os.path.isdir('saved_models_{}'.format('sngan')):
    os.mkdir('saved_models_{}'.format('sngan'))
f = open('saved_models_{}/log_collapse1.txt'.format('sngan'), mode='w')
import torch.utils.data as Data
import cv2
x = []
y = np.zeros((31, 1), dtype=np.int)
y = list(y)
for i in range(31):
   y[i] = []
# for Generative Adversarial Networks
# Ref:
# - https: // arxiv.org / abs / 1802.05957
# - https: // github.com / pfnet - research / sngan_projection / tree / master / source
import keras
from keras.datasets import cifar10
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

    model.add(Conv2D(3,(3,3),padding="same",activation="tanh"))
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
image_size = (32,32,3)
noise_size = (2,2,32)
batch_size = 64
sample_size=10
size=32
sample_interval=200
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_of_data = x_train.shape[0]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_test = (x_test.astype(np.float32) - 127.5) / 127.5
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

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
        sampleSize = 5000
        # If at save interval => save generated image samples
        if global_step % sample_interval == 0:
            x.append(epoch)
            gen_imgs = generator.predict(np.random.normal(size=(sample_size,) + noise_size))
            # gen_imgs = np.reshape(gen_imgs, (sample_size, sample_size, size, size, 3))
            # gen_imgs = np.transpose(gen_imgs, (0, 2, 1, 3, 4))
            # gen_imgs = np.reshape(gen_imgs, (sample_size * size, sample_size * size, 3))

            # gen_imgs = 255 * (gen_imgs + 1) / 2

            gen_imgs = np.array([cv2.resize(img, (299, 299)) for img in gen_imgs])
            X_test = np.array([cv2.resize(img, (299, 299)) for img in x_test[:sampleSize]])
            x_dataset = MyDataset(X_test)
            # print(x_dataset[0].shape)
            x_real_loader = Data.DataLoader(dataset=x_dataset, batch_size=64, shuffle=True)
            x_fake_dataset = MyDataset(gen_imgs)
            x_fake_loader = Data.DataLoader(dataset=x_fake_dataset, batch_size=64, shuffle=True)
            s = compute_score_raw(x_real_loader, x_fake_loader, 256, '/real/', './fake', conv_model='inception',
                                  workers=int(1))
            # real_images = tf.convert_to_tensor(X_test)  # real images
            # # MNIST_CLASSIFIER_FROZEN_GRAPH = '.\classify_mnist_graph_def.pb'
            # gen_imgs = np.array(gen_imgs)
            # eval_images = tf.convert_to_tensor(gen_imgs)
            # eval_score = utils.mnist_score(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)  # IS score
            # frechet_distance = utils.mnist_frechet_distance(real_images, eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)
            # mnist_score, f_distance = sess.run([eval_score, frechet_distance])
            # # print(mnist_score)
            # # print(f_distance)
            # # s[14]=mnist_score
            # # s[16]=f_distance
            # s[17]=mnist_score
            # s[18]=f_distance
            # print('IS socre: %f' % mnist_score)
            # print('FID: %f' % f_distance)

            for i in range(len(s)):
                print(i, "=", s[i])
            for i in range(len(s)):
                y[i].append(s[i])
            f.writelines('\n')
            f.writelines('epoch:' + str(epoch))
            f.writelines('\n')
            f.writelines(' %.8f ' % (i) for i in s)
            f.writelines('\n')
for i in range(len(s)):
    y[i] = [float(j) / max(y[i]) for j in y[i]]  # 对值进行归一化处理

for i in range(len(s)):
    font1 = {'size': 8}
    plt.plot(x, y[i], label=labels_name[i])
    plt.legend(loc='lower right', prop=font1)
    plt.savefig('saved_models_sngan/{}.png'.format(labels_name[i]))
    plt.show()
    plt.close()
