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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

def get_data():
    """
    function_word_list = [  '之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍', 
                            '故', '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀', 
                            '吗', '咧', '啊', '把', '让', '向', '往', '是', '在', '越', 
                            '再', '更', '比', '很', '偏', '别', '好', '可', '便', '就',
                            '但', '儿',                 # 42 个文言虚词
                            '又', '也', '都', '要',      # 高频副词
                            '这', '那', '你', '我', '他' # 高频代词
                            '来', '去', '道', '笑', '说' #高频动词
                            ]
    """
    function_word_list = [  '之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍', 
                            '故', '尚', '乃', '呀', '吗', '咧', '啊', '罢', '么', '呢',
                            '了', '的', '着', '一', '不', '把', '让', '向', '往', '是', 
                            '在', '别', '好', '可', '便', '就', '但', '越', '再', '更',
                            '比', '很', '偏', '罢咧', '罢了']

    text = open('text/redmansions.txt').read()
    reg = '(本章完)'
    chap = re.split(reg, text)
    chap = [i for i in chap if len(i) > 200]
    chap = chap[1:]

    data = np.zeros([len(chap), len(function_word_list)])
    for i in range(len(chap)):
        for j in range(len(function_word_list)):
            data[i, j]=chap[i].count(function_word_list[j])
    
    #这里使用的是该章中，该虚词在该章出现次数占该章统计的全部虚词出现的次数
    sdata = np.zeros([len(data), len(function_word_list)])
    for i in range(len(data)):
        for j in range(len(function_word_list)):
            sdata[i, j] = data[i,j] / sum(data[i])
    #归一化
    scaler = MinMaxScaler().fit(sdata)
    ndata = scaler.transform(sdata)
    return ndata

data = get_data()
kmeans = KMeans(n_clusters = 2, random_state = 0).fit(data)
print(kmeans.labels_)
X = range(1, 120)
Y = kmeans.labels_
plt.plot(X, Y, "o")
plt.plot([80,80], [0, 2], "-")
plt.show()