import os
from keras.models import Sequential, load_model
import numpy as np
model = load_model('./models/model.h5')
model.load_weights('./models/weights.best.hdf5')
def get_mean_var(y_logit):
    y_logit=np.abs(y_logit-0.1)
    return np.mean(y_logit,axis=1)

def get_possibility(images):
    labels = model.predict(np.array(images))
    print(labels)
    return labels
from keras.datasets import mnist
(X_train, y_train), (X_test, _) = mnist.load_data()
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
X_test = np.expand_dims(X_test, axis=3)
get_possibility(X_test[:10])
