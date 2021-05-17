#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Jingwen Zhong

"""

##
# Word2vec metric
import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")


def word2vec_metric(word1, word2):
    similarity_ = word_vectors.similarity(word1, word2)
    return word1, word2, similarity_


##
# read json file
# Trip lexicon
import json

with open('lex-ont.json', 'r') as myfile:
    data = myfile.read()
lexicon_dict = json.loads(data)

# Trip Ontology
import jsontrips

ontology_dict = jsontrips.ontology()


# recursive function to get all the parents
def parent_recursion(x, parents):
    try:
        y = ontology_dict[x]['parent']
        parents.append(y)
        y, count = parent_recursion(y, parents)
        return y, parents
    except:
        return y, parents


# Wu-Palmer metric on TRIPS ontology
def Wu_Palmer_metric(word1, word2):
    y1 = lexicon_dict[word1]['lf_parent']
    y2 = lexicon_dict[word2]['lf_parent']
    parents1 = []
    parents2 = []
    for i in range(len(y1)):
        root, parent = parent_recursion(y1[i], [y1[i]])
        parents1.append(parent)

    for i in range(len(y2)):
        root, parent = parent_recursion(y2[i], [y2[i]])
        parents2.append(parent)

    Depth_LCS = 0
    similarity_ =0
    for i in range(len(parents1)):
        for j in range(len(parents2)):
            common_root = list(set(parents1[i]).intersection(set(parents2[j])))
            if len(common_root) > Depth_LCS:
                Depth_LCS = len(common_root)
                Depth_word1 = len(parents1[i])
                Depth_word2 = len(parents2[j])
                similarity_ = 2 * Depth_LCS / (Depth_word1 + Depth_word2)
    return word1, word2, similarity_

##
# preprocessing Brown corpus
import re
import nltk

brown_lemmatized = open("brown_lemmatized.txt", "r")
Brown_corpus = []
for string in brown_lemmatized:
    new_string = re.sub(r"\W+|_", " ", string)
    word_token = nltk.word_tokenize(new_string)
    Brown_corpus += word_token

# lexicon
with open('trips-brown_NV_overlap.txt') as f:
    vals = [nltk.word_tokenize(line) for line in f]
    words_lexicon = [v[0] for v in vals]

##
import numpy as np


# word-vector approach to compute off the Brown corpus

def word_vector(word1, word2):
    v1 = np.zeros(len(words_lexicon))
    v2 = np.zeros(len(words_lexicon))

    for i in range(len(Brown_corpus)):
        if word1 == Brown_corpus[i]:
            for j in range(i - 4, i + 5):
                if Brown_corpus[j] in words_lexicon and j != i:
                    k = words_lexicon.index(Brown_corpus[j])
                    v1[k] += 1

        if word2 == Brown_corpus[i]:
            for j in range(i - 4, i + 5):
                if Brown_corpus[j] in words_lexicon and j != i:
                    k = words_lexicon.index(Brown_corpus[j])
                    v2[k] += 1

        # cosine matrix
    Lv1 = np.sqrt(v1.dot(v1))
    Lv2 = np.sqrt(v2.dot(v2))

    similarity_ = v1.dot(v2) / (Lv1 * Lv2)

    return word1, word2, similarity_


##
# My "novel" technique

# def forth_technique(word1, word2):
#     word1_1, word1_2, similarity1 = word2vec_metric(word1, word2)
#     word2_1, word2_2, similarity2 = Wu_Palmer_metric(word1.upper(), word2.upper())
#     word3_1, word3_2, similarity3 = word_vector(word1, word2)
#     similarity_ = [similarity1,similarity2,similarity3]
#     return word1, word2, max(similarity_)

from gensim.models import Word2Vec

brown_lemmatized = open("brown_lemmatized.txt", "r")
Brown_corpus2 = []

for string in brown_lemmatized:
    new_string = re.sub(r"\W+|_", " ", string)
    word_token =(new_string)
    Brown_corpus2.append(word_token)


def forth_technique(word1, word2):
    model = Word2Vec(sentences=Brown_corpus2, min_count=1)
    model.save("word2vec.model")
    word_vectors2 = model.wv
    word_vectors2.save("word2vec.wordvectors")
    similarity_ = word_vectors2.similarity(word1, word2)
    return word1, word2, similarity_

##
# output:
def predict(inputs):
    results = []

    for j in range(len(inputs)):
        S1 = S2 = S3 = S4 = 0
        w1_1 = w1_2 = w2_1 = w2_2 = w3_1 = w3_2 = w4_1 = w4_2 = 0

        line_new = nltk.word_tokenize(inputs[j])

        for i in range(len(line_new)):
            word1_1, word1_2, similarity1 = word2vec_metric(line_new[i - 1], line_new[i])
            if similarity1 > S1:
                S1 = similarity1
                w1_1 = word1_1
                w1_2 = word1_2

            word2_1, word2_2, similarity2 = Wu_Palmer_metric(line_new[i - 1].upper(), line_new[i].upper())
            if similarity2 > S2:
                S2 = similarity2
                w2_1 = word2_1.lower()
                w2_2 = word2_2.lower()

            word3_1, word3_2, similarity3 = word_vector(line_new[i - 1], line_new[i])
            if similarity3 > S3:
                S3 = similarity3
                w3_1 = word3_1
                w3_2 = word3_2

            word4_1, word4_2, similarity4 = forth_technique(line_new[i - 1], line_new[i])
            if similarity4 > S4:
                S4 = similarity4
                w4_1 = word4_1
                w4_2 = word4_2

        result = [w1_1, w1_2, S1, w2_1, w2_2, S2, w3_1, w3_2, S3, w4_1, w4_2, S4]
        results.append(result)

    return results


##
def main():
    import sys
    with open(sys.argv[1]) as f1:
        inputs = f1.read()
        inputs = list(inputs.split('\n'))

    with open(sys.argv[2], 'r+',) as f2:
        predicts = predict(inputs)
        print(predicts)
        for line in predicts:
            for i in line:
                f2.write(str(i)+' ')
            f2.write("\n")
        f2.close()


if __name__ == '__main__':
    main()
