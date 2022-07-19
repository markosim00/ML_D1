# TODO popuniti kodom za problem 4
#%tensorflow_version 1.x
#!pip install nltk
import nltk
nltk.download()

nltk.download('stopwords')
from nltk.stem import PorterStemmer    # najmanje agresivan
from nltk.stem import LancasterStemmer # najvise agresivan
from nltk.stem import SnowballStemmer
porter = PorterStemmer()

from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
from google.colab import drive
drive.mount('/content/drive')
import csv
import re
import operator

class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount
  
  def fit(self, X, Y):
    nb_examples = X.shape[0]

    # Racunamo P(Klasa) - priors
    # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
    # broja u intervalu [0, maksimalni broj u listi]
    self.priors = np.bincount(Y) / nb_examples
    print('Priors:')
    print(self.priors)

    # Racunamo broj pojavljivanja svake reci u svakoj klasi
    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):
      c = Y[i]
      for w in range(self.nb_words):
        cnt = X[i][w]
        occs[c][w] += cnt
    print('Occurences:')
    print(occs)
    
    # Racunamo P(Rec_i|Klasa) - likelihoods
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
    print('Likelihoods:')
    print(self.like)
          
  def predict(self, bow):
    # Racunamo P(Klasa|bow) za svaku klasu
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    # Trazimo klasu sa najvecom verovatnocom
    print('\"Probabilites\" for a test BoW (with log):')
    print(probs)
    prediction = np.argmax(probs)
    return prediction
  
  def predict_multiply(self, bow):
    # Racunamo P(Klasa|bow) za svaku klasu
    # Mnozimo i stepenujemo kako bismo uporedili rezultate sa slajdovima
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = self.priors[c]
      for w in range(self.nb_words):
        cnt = bow[w]
        prob *= self.like[c][w] ** cnt
      probs[c] = prob
    # Trazimo klasu sa najvecom verovatnocom
    print('\"Probabilities\" for a test BoW (without log):')
    print(probs)
    prediction = np.argmax(probs)
    return prediction

from nltk.corpus.reader.nkjp import XML_Tool
import csv


def Convert(lst):
    res_dct = {lst[i][0]: lst[i][1] for i in range(0, len(lst), 2)}
    return res_dct

arr = []
csv.field_size_limit(10000000)
filename = '/content/drive/My Drive/Colab Notebooks/fake_news.csv'
with open(filename, newline='') as file:
  reader = csv.reader(file)
  arr = list(reader)



clean_corpus = []
dataset = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))
i = 0
regexp = re.compile('[^a-zA-Z]+')

#skracujemo tekst na 100 elemenata
number_of_docs = 100


Y = np.zeros(number_of_docs-1, dtype=np.int32)

for doc_idx in range(1,number_of_docs):
  doc = ' '.join(arr[doc_idx][1:-1])
  Y[doc_idx-1] = arr[doc_idx][-1]
  words = wordpunct_tokenize(doc)
  words_lower = [w.lower() for w in words]
  words_filtered = [w for w in words_lower if w not in stop_punc]
  words_stemmed = [porter.stem(w) for w in words_filtered]
  words_stemmed = [w for w in words_stemmed if not regexp.search(w)]
  dataset.extend(words_stemmed)
  clean_corpus.append(words_stemmed)
 
  

# trazimo najcescih 10k reci iz celog skupa reci
dicti = {}
[dicti.setdefault(val, 0) for val in dataset] 
for val in dataset:
  dicti[val]+=1
dicti = sorted(dicti.items(),key=operator.itemgetter(1),reverse=True)
dicti = dicti[:10000]
dicti = Convert(dicti)


vocab_set = set()
for doc in clean_corpus:
  for word in doc:
    if(word in dicti):
      vocab_set.add(word)

vocab = list(vocab_set)


np.set_printoptions(precision=2, linewidth=200)
 
X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
to_delete = []

#trazimo dokumente koji imaju 0 elemenata i stavljamo njihove indekse u listu kako bi ih posle obrisali iz X i Y
for doc_idx in range(len(clean_corpus)):
  if(len(clean_corpus[doc_idx])==0):
    to_delete.append(doc_idx)

to_delete.sort(reverse=True)

for val in to_delete:
  clean_corpus.pop(val)
  np.delete(Y,val)


def occ_score(word, doc):
   return 1 if word in doc else 0
  
def numocc_score(word, doc):
  return doc.count(word)

def freq_score(word, doc):
  return doc.count(word) / len(doc)


for score_fn in [occ_score, numocc_score, freq_score]:
  for doc_idx in range(len(clean_corpus)):
    doc = clean_corpus[doc_idx]
    for word_idx in range(len(vocab)):
      word = vocab[word_idx]
      cnt = score_fn(word, doc)
      X[doc_idx][word_idx] = cnt

corpus = clean_corpus
labels = Y[:1000]
limit = math.floor(len(corpus) * 0.80)
train_corpus = corpus[:limit]
test_corpus = corpus[limit:]
train_labels = labels[:limit]
test_labels = labels[limit:]

model = MultinomialNaiveBayes(nb_classes = 2, nb_words = 1000, pseudocount = 1)
model.fit(X, Y)

def create_test_bow(doc, vocab):
    test_bow = np.zeros(len(vocab), dtype=np.float64)
    for word_idx in range(len(vocab)):
      word = vocab[word_idx]
      cnt = numocc_score(word, doc)
      test_bow[word_idx] = cnt
    return test_bow

matches = 0

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(0, len(test_corpus)):
  test_bow = create_test_bow(test_corpus[i], vocab)
  prediction = model.predict(test_bow)

  if prediction == 1:
    if test_labels[i] == prediction:
      TP += 1
      matches += 1
    else:
      FP += 1
  else:
    if test_labels[i] == prediction:
      TN += 1
      matches += 1
    else:
      FN += 1

accuracy = matches/len(test_corpus)
print('Accuracy: ', accuracy)

matrix_confusion = [[TN, FP], [FN, TP]]
print('Matrix confusion: ', matrix_confusion)

freq_dict_1 = dict()
freq_dict_0 = dict()

for i in range(len(corpus)):
  if labels[i] == 1:
    for word in corpus[i]:
      freq_dict_1.setdefault(word, 0)
      freq_dict_1[word] += 1

for i in range(len(corpus)):
  if labels[i] == 0:
    [freq_dict_0.setdefault(word, 0) for word in corpus[i]]
    for word in corpus[i]:
      freq_dict_0[word] += 1 

freq_dict_1 = sorted(freq_dict_1, key = freq_dict_1.get, reverse = True)[:5]
freq_dict_0 = sorted(freq_dict_0, key = freq_dict_0.get, reverse = True)[:5]

print('Most frequent words in class 1: ', freq_dict_1)
print('Most frequent words in class 0: ', freq_dict_0)

#Vidimo da se neke od najcesce koriscenih reci u fajlu uopste javljaju i kao
# najcesce koriscene reci u obe klase, tako da njihovo pojavljivanje ne zavisi
# puno od toga da li je clanak pouzdan ili nepouzdan

counter_0 = 0
counter_1 = 0


def LR(word):
  global counter_0
  global counter_1
  for i in range(len(corpus)):
    if labels[i] == 1:
      counter_1 += corpus[i].count(word)
    else:
      counter_0 += corpus[i].count(word)

  if counter_1 >= 10 and counter_0 >= 10:
    return counter_1/counter_0
  else:
    return 0

LR_array = dict()

for i in range(len(corpus)):
  for word in corpus[i]:
    LR_array[word] = LR(word)

lowest_LR = sorted(LR_array, key=LR_array.get)[:5]
highest_LR = sorted(LR_array, key=LR_array.get, reverse = True)[:5]

print('Words with lowest LR: ', lowest_LR)
print('Words with highest LR: ', highest_LR)
    
#LR metrika predstavlja srazmeru pojavljivanja reci u pouzdanim u odnosu na
# pojavljivanje recu u nepouzdanim clancima
# Iz dobijenih podataka vidimo da su reci sa najmanjom i najvecom LR metrikom 
# neka licna imena ili imena twitter naloga, sto znaci da nam LR metrika govori
# koji su pouzdani a koji nepouzdani izvori