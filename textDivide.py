import os
import re
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jieba
import jieba.posseg as pseg

def split_to_words(document, docID):
    pathStr = 'text/chapter-words-' + str(docID) + '.txt'
    file_out = open(pathStr, 'w')

    string_in = document.readline()
    while(string_in):
        cutWord = jieba.cut(string_in, cut_all=False)
        words = " ".join(cutWord)
        file_out.write(words)
        string_in = document.readline()
    file_out.close()

def words_count(document, docID):
    result_dict = {}
    delset = string.punctuation

    line = str(document)
    line = line.translate(delset) #去除英文标点
    line = "".join(line.split('\n')) # 去除回车
    line = replace_rule(line) #去除中文标点
    words = line.split()
    for word in words:
        if word not in result_dict:
            result_dict[word] = 1
        else:
            result_dict[word] += 1

    pathStr = 'text/chapter-wordcount-' + str(docID) + '.txt'
    file_out = open(pathStr,'w')

    sorted_result = sorted(result_dict.items(), key=lambda d:d[1], reverse = True)
    for one in sorted_result:
        line = "".join(one[0] + '\t' + str(one[1]) + '\n')
        file_out.write(line)
    file_out.close()

def tag_processor(docID):
    pathStr = 'text/chapter-' + str(docID) + '.txt'
    file_in = open(pathStr, 'r')
    document = ' '
    file_in.readline()
    for line in file_in.readlines():
        line.strip ()
        document += line
    document = ''.join(document.split('\n'))
    document = pseg.cut(document)
    file_out = open('text/word-tag-' + str(docID) + '.txt', 'w')
    result_dict = {}
    for term in document:
        if term.flag not in result_dict:
            result_dict[term.flag] = 1
        else:
            result_dict[term.flag] += 1
    terms = result_dict.items()
    for one in terms:
        file_out.write(one[0] + '\t' + str (one[1]) + '\n') 
    file_in.close()
    file_out.close()
    print('chapter ' + str(docID) + ' finish')

def replace_rule(line):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]")
    line = rule.sub('',line)
    return line

# Divide into chapters
file_in = open('redmansions.txt', 'r')
string_in = file_in.readline()
chapterNum = 0
chapterText = ""
pathStr = ""
while string_in:
    if '(本章完)' in string_in:
        chapterNum += 1
        pathStr = 'text/chapter-' + str(chapterNum) + '.txt'
        file_out = open(pathStr, 'w')
        file_out.write(chapterText)
        file_out.close()
        chapterText = ""
    else:
        chapterText += string_in
    string_in = file_in.readline()
file_in.close

# Divide each chapter to words
for loop in range(1, 121):
    pathStr = 'text/chapter-' + str(loop) + '.txt'
    file_in = open(pathStr, 'r')
    split_to_words(file_in, loop)

# Count the number
for loop in range(1, 121):
    pathStr = 'text/chapter-words-' + str(loop) + '.txt'
    file_in = open(pathStr, 'r')
    line = file_in.readline()
    document = ""
    while line:
        document += line
        line = file_in.readline()
    words_count(document, loop)
    file_in.close()

# Part of speech
for loop in range(1, 121):
    tag_processor(loop)