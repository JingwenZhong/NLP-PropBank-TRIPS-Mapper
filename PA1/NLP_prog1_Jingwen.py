# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:43:59 2021

@author: Jingwen
"""

##
import numpy as np
import nltk
from nltk.corpus import brown
from nltk import ngrams
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors

lemmatizer = WordNetLemmatizer()
colors_set = list(mcolors.CSS4_COLORS)  # A set of words for colors


# this includes 148 colors


##
def predict(color, objects):  # inputs: color and objects
    # "color" and "objects" are 2 lists
    tokens = brown.words()  # read all the words in brown corpus
    tokens_new = []
    for i in range(len(tokens)):
        tokens_new.append(lemmatizer.lemmatize(tokens[i]))
    mygrams = ngrams(tokens_new, 2)
    fdist = FreqDist(mygrams)  # calculated frequency of bigrams from brown corpus
    # and this is a dictionary with bigram pairs and their frequency.

    candidates = []  # put all the satisfied pairs and its frequency into this candidates
    predicts = []
    extra = []
    s = 0
    # start the algorithm
    for i in range(len(objects)):  # everything below is under this for loop
        for k, v in fdist.items():  # k is the pair, v is frequency
            # k[0] is the first word of the pair and we expect it to be a color word
            # k[1] is the second word of the pair and we expect it matches our objects

            # Testing 1
            if k[1] == objects[i]:  # if the input object[i] is in brown corpus
                if k[0] in colors_set and color[i] in set(k[0]):
                    # if k[0] is a color word and input color[i] is in this color word set
                    candidates.append((k, v))  # it is a candidate
            else:  # else just put the original inputs and assign its frequency as 1
                candidates.append(((color[i], objects[i]), 1))
        candidates_ = dict(candidates)  # convert to dictionary
        a = zip(candidates_.values(), candidates_.keys())
        b = list(max(a)[1])  # find the pair that as the most frequency and this will be the predicted output

        # Bonus:deal with the words that is not in brown corpus or it has not color matched pair in brown corpus
        if b[
            0] not in colors_set:  # b[0] is color[i] which is a letter, if b[0] is not equal to any of those color words
            for C in colors_set:  # color word
                for k, v in fdist.items():
                    if b[0] in set(C) and k[0] == C:  # if b[0] is in C, and this C shows up in brown corpus.
                        s += v  # add up the frequency of C that shows up in brown corpus
                extra.append((C, s))
                s = 0
            extra_ = dict(extra)
            d = zip(extra_.values(), extra_.keys())
            e = max(d)[1]  # find the C that has most frequency
            b[0] = e  # update b[0]

        predicts.append(' '.join(b))  # final predictions
        candidates.clear()  # clean the candidates set

    return predicts


##
def parse_line(line):  # word_token and lemmatizer all the lines
    line_split = nltk.word_tokenize(line)
    line_new = []
    for i in range(len(line_split)):
        line_new.append(lemmatizer.lemmatize(line_split[i]))
    return line_new


def main():
    import sys
    with open(sys.argv[1]) as f:
        # separate the first word and the second word in each line in the text file
        # and put all the first words together as an array, put all the second words in another list together as an array
        vals = [parse_line(line) for line in f]
        (color, objects) = ([v[0] for v in vals], [v[1] for v in vals])
        np.asarray(color)
        np.asarray(objects)

    with open(sys.argv[2]) as f2: # output
        contents = f2.read()
        predicts = predict(color, objects)
        print(predicts)
        my_file = open((sys.argv[2]), "r+")
        for phrase in range(len(predicts)):
            my_file.write(str(phrase) + "\n")
        my_file.close()


if __name__ == '__main__':
    main()
