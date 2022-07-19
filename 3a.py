# TODO popuniti kodom za problem 3a
#%tensorflow_version 1

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
import csv 
#%matplotlib inline

class KNN:
  
  def __init__(self, nb_features, nb_classes, data, k):
    self.nb_features = nb_features
    self.nb_classes = nb_classes
    self.data = data
    self.k = k
    
    # Gradimo model, X je matrica podataka a Q je vektor koji predstavlja upit.
    self.X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
    self.Q = tf.placeholder(shape=(None), dtype=tf.float32)
    self.predicts = []
    
    # Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)), 
                                  axis=1))
    _, idxs = tf.nn.top_k(-dists, self.k)  
    
    self.classes = tf.gather(self.Y, idxs)
    self.dists = tf.gather(dists, idxs)
    
    self.w = tf.fill([k], 1/k)
    
    w_col = tf.reshape(self.w, (k, 1))
    self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
    self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)
    
    # Klasa sa najvise glasova je hipoteza.
    self.hyp = tf.argmax(self.scores)
   
  # Ako imamo odgovore za upit racunamo i accuracy.

  def predict(self, query_data):
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
     
      #nb_queries = query_data['x'].shape[0]
      
      # Pokretanje na svih 10000 primera bi trajalo predugo,
      # pa pokrecemo samo prvih 100.
      nb_queries = 100
      
      matches = 0
      for i in range(nb_queries):
        hyp_val = sess.run(self.hyp, feed_dict = {self.X: self.data['x'], 
                                                  self.Y: self.data['y'], 
                                                 self.Q: query_data['x'][i]})
        if query_data['y'] is not None:
          actual = query_data['y'][i]
          match = (hyp_val == actual)
          if match:
            matches += 1
          if i % 2 == 0:
            print('Test example: {}/{}| Predicted: {}| Actual: {}| Match: {}'
                 .format(i+1, nb_queries, hyp_val, actual, match))
      
      accuracy = matches / nb_queries
      print('{} matches out of {} examples'.format(matches, nb_queries))
      return accuracy

  def predictions(self, query_data):
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      nb_queries = len(query_data)

      matches = 0
      for i in range(nb_queries):
        hyp_val = sess.run(self.hyp, feed_dict = {self.X: self.data['x'],
                                                  self.Y: self.data['y'],
                                                  self.Q: query_data['x'][i]})
        self.predicts.append(hyp_val)

    return np.asarray(self.predicts)


with open('social_network_ads.csv', newline='') as file:
    reader = csv.reader(file)
    all_data = list(reader)

all_data.pop(0)

print(len(all_data))

data = {'x': [], 'y': []}

for ad in all_data:
    list_x = list()
    for i in range(2,5):
        if i < 4:
            list_x.append(ad[i])
        else:
          if ad[i] == '0':
            data['y'].append(0)
          elif ad[i] == '1':
            data['y'].append(1)
    data['x'].append(list_x)


print(len(data['x']))
print(len(data['y']))

print(data['x'])
print(data['y'])

train_x = []
train_y = []

test_x = []
test_y = []


for i in range(0, len(data['x'])):
  if random.random() <= 0.7:
    train_x.append(data['x'][i])
    train_y.append(data['y'][i])
  else:
    test_x.append(data['x'][i])
    test_y.append(data['y'][i])


nb_train = len(train_y)
nb_test = len(test_y)
print(train_y)

train_x = np.reshape(train_x, [nb_train, -1])
test_x = np.reshape(test_x, [nb_test, -1])

nb_features = 2
nb_classes = 2
k = 3
train_data = {'x': train_x, 'y': train_y}
knn = KNN(nb_features, nb_classes, train_data, k)
accuracy = knn.predict({'x': test_x, 'y': test_y})
print('Test set accuracy: ', accuracy)

data_for_plot_00 = []
data_for_plot_01 = []
data_for_plot_10 = []
data_for_plot_11 = []


for i in range(0, nb_train):
  if train_y[i] == 0:
    data_for_plot_00.append(train_x[i][0])
    data_for_plot_01.append(train_x[i][1])
  elif train_y[i] == 1:
    data_for_plot_10.append(train_x[i][0])
    data_for_plot_11.append(train_x[i][1])


plt.scatter(data_for_plot_00, data_for_plot_01, c = 'b', label = 'Klasa 0')
plt.scatter(data_for_plot_10, data_for_plot_11, c = 'r', label = 'Klasa 1')


plt.legend()
plt.show()

