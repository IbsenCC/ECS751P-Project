import os
import re
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm

def svm_learn(train_data, train_data_label, predict_data):
    output = svm.SVC()
    output.fit(train_data, train_data_label)
    result = output.predict(predict_data)
    return result

trainset = np.load('trainset.npy')
testset = np.load('testset.npy')

train_data = trainset[:, 0:-1]
train_data_label = trainset[:, -1]
test_data = testset[:, 0:-1]

result = svm_learn(train_data, train_data_label, test_data)
print (len(result))
print (result[0:80])
print (result[80:120])

X = range(1, 121)
Y = result
plt.plot(X, Y, "o")
plt.plot([80,80], [0, 2], "-")
plt.show()