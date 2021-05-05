# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QbzMxaf2zlugQiqzGEXXAA6d7GYU1Ksm
"""

cd sample_data/

x = list()
y = list()
f = open('spambase.data', 'r')
while True:
  data = f.readline()
  if data == '' :
    break
  data_split = data.split(',')
  if data_split[-1][0] == '1':
    y.append(1)
  else:
    y.append(0)
  data_split.pop()
  int_data = list(map(float, data_split))
  x.append(int_data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, random_state = 2021)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(solver='adam', hidden_layer_sizes=10, activation='relu', alpha=0.0001, max_iter=200, random_state=1016)
mlp.fit(x_train_sc, y_train)

y_train_hat = mlp.predict(x_train_sc)
y_test_hat = mlp.predict(x_test_sc)

train_accuracy = accuracy_score(y_train, y_train_hat)
test_accuracy = accuracy_score(y_test, y_test_hat)

print('train acc : ', train_accuracy)
print('test acc : ', test_accuracy)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
hidden_layer = [450, 500, 550, 600]
for hl in hidden_layer:
  print('hidden layer : ', hl)
  mlp = MLPClassifier(hidden_layer_sizes=hl, random_state=1016, max_iter = 600)
  mlp.fit(x_train_sc, y_train)

  y_train_hat = mlp.predict(x_train_sc)
  y_test_hat = mlp.predict(x_test_sc)

  train_accuracy = accuracy_score(y_train, y_train_hat)
  test_accuracy = accuracy_score(y_test, y_test_hat)

  print('train acc : ', train_accuracy)
  print('test acc : ', test_accuracy)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
solver = ['lbfgs', 'sgd', 'adam']
for sv in solver:
  print('solver : ', sv)
  mlp = MLPClassifier(hidden_layer_sizes=500, solver=sv, random_state=1016, max_iter=600)
  mlp.fit(x_train_sc, y_train)

  y_train_hat = mlp.predict(x_train_sc)
  y_test_hat = mlp.predict(x_test_sc)

  train_accuracy = accuracy_score(y_train, y_train_hat)
  test_accuracy = accuracy_score(y_test, y_test_hat)

  print('train acc : ', train_accuracy)
  print('test acc : ', test_accuracy)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
activation = ['relu', 'tanh']
for ac in activation:
  print('activation : ', ac)
  mlp = MLPClassifier(hidden_layer_sizes=500, solver='adam', random_state=1016, max_iter=600, activation=ac)
  mlp.fit(x_train_sc, y_train)

  y_train_hat = mlp.predict(x_train_sc)
  y_test_hat = mlp.predict(x_test_sc)

  train_accuracy = accuracy_score(y_train, y_train_hat)
  test_accuracy = accuracy_score(y_test, y_test_hat)

  print('train acc : ', train_accuracy)
  print('test acc : ', test_accuracy)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
alpha = [0.0001, 0.001, 0.01, 0.1]
for ap in alpha:
  print('alpha : ', ap)
  mlp = MLPClassifier(hidden_layer_sizes=500, solver='adam', random_state=1016, max_iter=600, activation='relu', alpha=ap)
  mlp.fit(x_train_sc, y_train)

  y_train_hat = mlp.predict(x_train_sc)
  y_test_hat = mlp.predict(x_test_sc)

  train_accuracy = accuracy_score(y_train, y_train_hat)
  test_accuracy = accuracy_score(y_test, y_test_hat)

  print('train acc : ', train_accuracy)
  print('test acc : ', test_accuracy)

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler()
mmscaler.fit(x_train)
x_train_sc = mmscaler.transform(x_train)
x_test_sc = mmscaler.transform(x_test)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(hidden_layer_sizes=500, solver='adam', random_state=1016, max_iter=600, activation='relu', alpha=0.001)
mlp.fit(x_train_sc, y_train)

y_train_hat = mlp.predict(x_train_sc)
y_test_hat = mlp.predict(x_test_sc)

train_accuracy = accuracy_score(y_train, y_train_hat)
test_accuracy = accuracy_score(y_test, y_test_hat)

print('train acc : ', train_accuracy)
print('test acc : ', test_accuracy)