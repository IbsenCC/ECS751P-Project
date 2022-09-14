import os
import re
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jieba
import jieba.analyse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

text = open('redmansions.txt').read()
reg = '(本章完)'
chap = re.split(reg, text)
chap = [i for i in chap if len(i) > 200]
index = range(1, 121)
result = pd.DataFrame({'id':index,'text':chap})

vorc = [jieba.analyse.extract_tags(i, topK = 1000) for i in result["text"]]
vorc = [" ".join(i) for i in vorc]

vectorizer = CountVectorizer(max_features = 2000)
train_data_features = vectorizer.fit_transform(vorc)
train_data_features = train_data_features.toarray()

print(vectorizer.get_feature_names(), len(vectorizer.get_feature_names()))

kmeans = KMeans(n_clusters = 2, random_state = 0).fit(train_data_features[0:120])
print(len(train_data_features) + 1)
print(kmeans.labels_)
X = range(1, len(train_data_features) + 1)
Y = kmeans.labels_
plt.plot(X, Y, "o")
plt.plot([80,80], [0, 2], "-")
plt.show()

cluster = 8

f = plt.figure(figsize=(20, 5))
for i in range(2, cluster + 1):
    subplot = "1" + str(cluster) + str(i)
    kmeans = KMeans(n_clusters = i, random_state = 0).fit(train_data_features[0:120])
#     print(kmeans.labels_)

    X = range(1, len(train_data_features) + 1)
    Y = kmeans.labels_
    ax = f.add_subplot(int(subplot))
#     plt.subplot(subplot)

    plt.plot(X, Y, "o")
    plt.plot([80,80], [0, cluster], "-")
#     plt.subplot()