# TODO popuniti kodom za problem 3b
#%tensorflow_version 1.x

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import random 
#%matplotlib inline

tf.reset_default_graph()

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

with open('social_network_ads.csv', newline='') as file:
    reader = csv.reader(file)
    all_data = list(reader)

all_data.pop(0)

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

nb_train_x = len(train_x)
nb_test_x = len(test_x)
nb_train_y = len(train_y)
nb_test_y = len(test_y)

print(nb_train_x)
print(nb_test_x)
print(nb_train_y)
print(nb_test_y)

nb_features = 2
nb_classes = 2
train_data = {'x': train_x, 'y': train_y}

k_diff = []
all_accuracy = []

for k in range(1, 16):
  knn = KNN(nb_features, nb_classes, train_data, k)
  accuracy = knn.predict({'x': test_x, 'y': test_y})
  k_diff.append(k)
  all_accuracy.append(accuracy)

fig, drawing = plt.subplots()

drawing.plot(k_diff, all_accuracy)
drawing.set(xlabel='k', ylabel='Accuracy', title='Dependence Accuracy of value k')
drawing.grid()
plt.show()

# Funkcija se cesto ponasa periodicno u odredjenim intervalima, najcesce je za 
# neparne vrednosti k manja nego za parne, najbolji rezultat se najcesce dobija
# za vrednosti 8, 10, 12, mada nije pravilo,
# accuracy je najcesce u intervalu od 0.8 do 0.9, ili 0.75 do 0.85,
# nema velikih promena accuracy-a, retko je interval siri od 0.1

