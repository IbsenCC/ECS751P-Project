import os
import re
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_wordnum_of_chapter(DocID):
    pathStr = 'text/chapter-' + str(DocID) + '.txt'
    file_in = open(pathStr, 'r')

    text = ""
    for string_in in file_in:
        text += "".join(string_in.split('\n')) # 去除回车
    file_in.close

    result = len(text)
    return result

# 用词性提取特征向量
def build_feature_vector(DocID, label):
    pathStr = 'text/word-tag-' + str(DocID) + '.txt'
    file_in = open(pathStr, 'r')

    feature_word_list = [  'tg', 'an', 'df', 'j', 'g', 'ng', 'rz', 'ud',
                            'k', 'ad', 'dg', 'e', 'y', 'nz', 'o', 'zg', 'i',
                            'uj', 'c', 'r', 'a', 'b', 'z', 'p', 'v', 'n', 'l',
                            'vn', 'u', 'nr', 'f', 'uz', 'mq', 'm', 't', 'ag', 'ul',
                            's', 'h', 'ns'
                            ]
        
    document = ''
    feature_vector_list = []

    for string_in in file_in.readlines():
        document += string_in

    document = document.split('\n')
    length = len(document)

    for i in range(0, length):
        document[i] = document[i].split('\t')

    for tag in feature_word_list:
        flag = 1
        for i in range(0, length):
            if tag == document[i][0]:
                total_words = get_wordnum_of_chapter(DocID)
                rate = float(int(document[i][1])) / total_words * 1000
                rate = float("%.6f" % rate)
                feature_vector_list.append(rate)
                flag = 0
                break
        if flag == 1:
            feature_vector_list.append(0)

    feature_vector_list.append(label)
    file_in.close()
    return feature_vector_list

# 用虚词提取特征向量
def build_feature_vector_function(DocID, label):
    pathStr = 'text/chapter-wordcount-' + str(DocID) + '.txt'
    feature_word_list = [  '之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍', 
                            '故', '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀', 
                            '吗', '咧', '啊', '把', '让', '向', '往', '是', '在', '越', 
                            '再', '更', '比', '很', '偏', '别', '好', '可', '便', '就',
                            '但', '儿', # 虚词 dummy words
                            '又', '也', '都', '要',      # 副词
                            '这', '那', '你', '我', '他' # 代词 pronoun
                            '来', '去', '道', '笑', '说' # 动词 verb
                            ] 
    feature_vector_list = []
        
    for function_word in feature_word_list:
        find_flag = 0
        file_in = open(pathStr) #每次打开移动 cursor 到头部
        string_in = file_in.readline()
        while string_in:
            words = string_in[:-1].split('\t')
            if words[0] == function_word:
                total_words = get_wordnum_of_chapter(DocID)
                rate = float(words[1]) / total_words * 1000
                rate = float("%.6f" % rate)# 指定位数
                feature_vector_list.append(rate)
                # print words[0] + ' : ' + string_in

                file_in.close()
                find_flag = 1
                break
            string_in = file_in.readline()

        # 未找到词时向量为 0
        if not find_flag:
            feature_vector_list.append(0) 
        
    feature_vector_list.append(label)
    return feature_vector_list

def make_positive_trainset():
    positive_trainset_list = []
    for loop in range(40, 60):
        feature = build_feature_vector(loop, 1) # label 为 1 表示正例
        positive_trainset_list.append(feature)
    np.save('pos_trainset.npy', positive_trainset_list)

def make_negative_trainset():
    negative_trainset_list = []
    for loop in range(100, 120):
        feature = build_feature_vector(loop, 0) # label 为 0 表示负例
        negative_trainset_list.append(feature)
    np.save('neg_trainset.npy', negative_trainset_list)

def make_positive_trainset_function():
    positive_trainset_list = []
    for loop in range(40, 60):
        feature = build_feature_vector_function(loop, 1) # label 为 1 表示正例
        positive_trainset_list.append(feature)
    np.save('pos_trainset.npy', positive_trainset_list)

def make_negative_trainset_function():
    negative_trainset_list = []
    for loop in range(100, 120):
        feature = build_feature_vector_function(loop, 0) # label 为 0 表示负例
        negative_trainset_list.append(feature)
    np.save('neg_trainset.npy', negative_trainset_list)

def make_trainset():
    feature_pos = np.load('pos_trainset.npy')
    feature_neg = np.load('neg_trainset.npy')
    trainset = np.vstack((feature_pos, feature_neg))
    np.save('trainset.npy', trainset)

def make_testset():
    testset_list = []
    for loop in range(1, 121):
        feature = build_feature_vector(loop, 0) # 无需 label，暂设为 0
        testset_list.append(feature)
    # print testset_list
    np.save('testset.npy', testset_list)

def make_testset_function():
    testset_list = []
    for loop in range(1, 121):
        feature = build_feature_vector_function(loop, 0) # 无需 label，暂设为 0
        testset_list.append(feature)
    # print testset_list
    np.save('testset.npy', testset_list)

'''
make_positive_trainset() 	
make_negative_trainset()
make_trainset()
make_testset()
'''

make_positive_trainset_function() 	
make_negative_trainset_function()
make_trainset()
make_testset_function()
