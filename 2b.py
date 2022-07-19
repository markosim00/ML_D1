%tensorflow_version 1.x
!pip install --upgrade xlrd

class bcolors:
  colors = [ '\033[95m','\033[94m','\033[96m']

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import sqlite3
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import matplotlib.pyplot as plt
import random


def create_plot_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)


def create_feature_matrix(x, nb_features, id):
    tmp_features = []
    features = random.randint(2, nb_features + 1)
    for deg in range(1, features):
        tmp_features.append(np.power(x, deg))
    if (values['k'] == 0):
        values[id] = features
    return np.column_stack(tmp_features)


filename = '/content/drive/My Drive/Colab Notebooks/2a.xls'
all_data = pd.read_excel(filename, usecols=(1, 2, 3, 4, 5, 6, 7))
data = dict()

data['x'] = all_data['temperature'][1000]
data['z'] = all_data['humidity'][1000]
data['k'] = all_data['lowcost_pm2_5'][1000]

values = {'x': 0, 'z': 0, 'k': 0}
data['y'] = all_data['reference_pm2_5'][1000]

nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)

data['x'] = data['x'][indices]
data['y'] = data['y'][indices]
data['z'] = data['z'][indices]
data['k'] = data['k'][indices]

data['y'] = data['y'].str.replace(',', '.').astype(float)

data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['z'] = (data['z'] - np.mean(data['z'], axis=0)) / np.std(data['z'], axis=0)
data['k'] = (data['k'] - np.mean(data['k'], axis=0)) / np.std(data['k'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

nb_features = 6

data['x'] = create_feature_matrix(data['x'], nb_features, 'x')
data['z'] = create_feature_matrix(data['z'], nb_features, 'z')
data['k'] = create_feature_matrix(data['k'], nb_features, 'k')

plt.scatter(data['x'][:, 0], data['y'])
plt.scatter(data['z'][:, 0], data['y'])
plt.scatter(data['k'][:, 0], data['y'])

loss_arr = []
degree_arr = []

for val in values:
    X = tf.placeholder(shape=(None, values[val] - 1), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    w = tf.Variable(tf.zeros(values[val] - 1))
    bias = tf.Variable(0.0)

    w_col = tf.reshape(w, (values[val] - 1, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)

    Y_col = tf.reshape(Y, (-1, 1))

    loss = tf.reduce_mean(tf.square(hyp - Y_col))

    opt_op = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        nb_epochs = 100

        for epoch in range(nb_epochs):
            final_loss = 0
            for sample in range(nb_samples):
                feed = {X: data[val][sample].reshape((1, values[val] - 1)),
                        Y: data['y'][sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)

            loss_arr.append(curr_loss)
            degree_arr.append(values[val] - 1)

        w_val = sess.run(w)
        bias_val = sess.run(bias)
        xs = create_plot_feature_matrix(np.linspace(-3, 4, 100), values[val] - 1)
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        plt.plot(xs[:, 0].tolist(), hyp_val.tolist())
        plt.xlim([-3, 3])
        plt.ylim([-2, 2])

plt.show()

xpoints = np.array(loss_arr)
ypoints = np.array(degree_arr)

plt.plot(xpoints, ypoints)
plt.show()


