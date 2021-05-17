#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Jingwen
"""

###
# Get the information from PropBank
import os
from nltk.corpus import propbank
import numpy as np


# get all the Propbank .xml files
def Files(file_dir):
    allfiles = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                allfiles.append(os.path.join(root, file))
    ffiles = []
    # only verb's file will be added
    for verb in propbank.verbs():
        for ff in allfiles:
            if verb in ff.split('\\')[-1]:  # make the running time shorter
                ffiles.append(ff)
    ffiles = np.unique(ffiles)
    return ffiles


from xml.dom.minidom import parse
from nltk.corpus import verbnet as vn
import string


# extract information from propbank frames
def PropbanK(file, word):
    domTree = parse(file)
    rootNode = domTree.documentElement
    predicates = rootNode.getElementsByTagName("predicate")
    for predicate in predicates:
        words = predicate.getAttribute("lemma")
        if words in file:
            propbank_ = []
            roleset1 = predicate.getElementsByTagName("roleset")
            for role in roleset1:
                elem = []  # store all the entries
                if role.hasAttribute("id"):
                    # append roleid eg. come.01
                    elem.append(role.getAttribute("id"))
                    # append role explanation eg. motion
                    elem.append(role.getAttribute("name"))

                    roles = role.getElementsByTagName("roles")
                    for element in roles:
                        r = element.getElementsByTagName("role")
                        # if the enrty aligns with verbnet, get the verbnet entry
                        try:
                            v = r[0].getElementsByTagName("vnrole")
                            if len(v) > 0:
                                vnrole = v[0].getAttribute("vncls")
                            else:
                                vnrole = ""
                        except:
                            pass

                    # get the wordnet entry that verbnet provides
                    vn_synset = []
                    vn_classid = vn.classids(word)
                    if len(vnrole) > 0:
                        for VN in vn_classid:
                            if vnrole in VN:
                                for j in vn.wordnetids(VN):
                                    j = j + "::"
                                    if '?' not in j:
                                        vn_synset.append(j)
                        elem.append(vn_synset)  # append the wordnet sense keys that verbnet provides
                    else:
                        elem.append(vn_synset)

                    # examples (this is not used)
                    Sentences = []
                    example = role.getElementsByTagName("example")
                    for element in example:
                        text = element.getElementsByTagName("text")[0]
                        if len(text.childNodes) > 0:
                            if '[' and ']' and '*' and "-" not in text.childNodes[0].data:  # delete some of the
                                sentence = text.childNodes[0].data.lower()
                                remove = str.maketrans('', '', string.punctuation)
                                sentence = sentence.translate(remove)
                                Sentences.append(sentence)
                    elem.append(Sentences)  # append examples

                propbank_.append(elem)

            return propbank_


def find_propbank(word):  # get the propbank entry for the word
    word1 = word + '.xml'
    files = Files('./frames')
    for file in files:
        if word1 == file.split('\\')[-1]:
            PROPBANK = PropbanK(file, word)
    return PROPBANK


###
# Get the information from TRIPS
from pytrips.ontology import load

ont = load()


def word_ont(name):
    name = name.lower()
    name = "w::{}".format(name)  # change the format of input to w::name
    word_ontology = ont[name, 'v']  # find all entries
    return word_ontology


import re
from nltk.corpus import wordnet as wn
import json


def find_trips(word):
    word_ontology = word_ont(word)

    # using collie-wnkey file to get the expanation and examples of the word sense key in wordnet
    with open('collie-wnkey.json', 'r') as myfile:
        data = myfile.read()
    wnkey_dict = json.loads(data)

    entries = []  # entries

    # try to find correct entries that are on the web browser
    for item in word_ontology:
        all_names = []
        for i in ont[item].wordnet_keys:
            for j in i.lemmas():
                if j.name() == word:
                    entries.append([item])
                all_names.append(j.name())

        # If none of "come" and "come" + preposition is found, then that means the type is directly mapped from the
        # lexicon
        for name in all_names:
            if word in name:
                aaa = []
                break
            else:
                aaa = ['a']
        if len(aaa) > 0:
            entries.append([item])  # eg. ONT::become

    entries = np.unique(entries)

    trips = []
    for item in entries:
        trips.append([item])

    # synset
    k = 0
    for item in entries:
        synset = []
        for i in ont[item].wordnet_keys:
            for j in i.lemmas():
                synset.append(re.sub('_', " ", j.name()))

        # wordnet key:
        wordnetkey = []
        for i in ont[item].wordnet:
            wordnetkey.append(i)

        wn_defin = []  # wordnet difinition
        wn_exam = []  # wordnet example (this is not used)
        item = str(item).upper()
        try:
            for i in ont[item].wordnet:
                for j in wnkey_dict[item]:
                    lemmas = wn.lemma_from_key(j)
                    synset.append(re.sub('_', " ", lemmas.name()))  # add more synset from collie-wnkey file
                    wordnetkey.append(j)  # add more wordnet key from collie-wnkey file
                    if i == j:
                        wn_defin.append(wnkey_dict[item][j]['definition'])  # wordnet difinition
                    if lemmas.name() == word:
                        wn_exam.append(wnkey_dict[item][j]['example'])  # wordnet example
        except:
            pass

        wn_defin = list(np.unique(wn_defin))
        wn_exam = list(np.unique(wn_exam))
        synset = list(np.unique(synset))

        # remove synset word that is equal to the input word
        for i in synset[::-1]:
            if i == word:
                synset.remove(i)

        wordnetkey = list(np.unique(wordnetkey))
        trips[k].append(synset)
        trips[k].append(wordnetkey)
        trips[k].append(wn_defin)
        trips[k].append(wn_exam)

        k += 1

    return trips


###
# Preparation functions:

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


def deal_text(text):  # (This function is not really used later)
    lower = text.lower()  # lower all letters

    remove1 = str.maketrans('', '', string.punctuation)  # store punctuations
    without_punctuation = lower.translate(remove1)  # remove punctuation

    tokens = nltk.word_tokenize(without_punctuation)  # word token
    without_stopwords = [w for w in tokens if not w in stopwords.words('english')]

    # delete the word that is not in wn.synsets ie. not a English word
    for w in without_stopwords[::-1]:
        if not wn.synsets(w):
            without_stopwords.remove(w)

    lemmatizer = WordNetLemmatizer()  # lemmatize words
    cleaned_text = [lemmatizer.lemmatize(ws, "v") for ws in without_stopwords]

    cleaned_text = list(np.unique(cleaned_text))

    return cleaned_text


import jsontrips

ontology_dict = jsontrips.ontology()


# get the path
def parent_recursion(x, parents):
    try:
        y = ontology_dict[x]['parent']
        parents.append(y)
        y, parents = parent_recursion(y, parents)
        return parents
    except:
        return parents


# get the similarity using wu palmer metric I wrote in PA2
def Wu_Palmer_metric(ONT1, ONT2):
    parents1 = parent_recursion(ONT1, [ONT1])
    parents2 = parent_recursion(ONT2, [ONT2])

    Depth_LCS = 0
    Similarity = 0

    common_root = list(set(parents1).intersection(set(parents2)))
    if len(common_root) > Depth_LCS:
        Depth_LCS = len(common_root)
        Depth_word1 = len(parents1)
        Depth_word2 = len(parents2)
        Similarity = 2 * Depth_LCS / (Depth_word1 + Depth_word2)

    return Similarity


from nltk import ngrams


# words combination of a sentence (used for 4th step of finding candidates process)
def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]


# find the third value (later will be score)
def takesecond(elem):
    return elem[2]


# seperate the sentence by punctuations
def split_sentences(line):
    line_split = re.split(r'[,():]', line.strip())
    line_split = [line.strip() for line in line_split if
                  line.strip() not in ['(', ')', ',', ":"] and len(line.strip()) > 1]
    return line_split


### Aligning

# get all the candidates with their aligning score
def candidates_func(word):
    propbank = find_propbank(word)  # get propbank information
    trips = find_trips(word)  # get trips information
    candidates = []

    for elem in propbank[::-1]:  # Outer loop PropBank
        pb_id = elem[0]
        pb_explain0 = elem[1]
        pb_explain = deal_text(pb_explain0)
        pb_explain1 = split_sentences(pb_explain0)  # use the split_sentences function
        pb_vn_entry = set(elem[2])
        # pd_example = elem[3]

        for i in pb_explain[::-1]:
            if i == word:
                pb_explain.remove(i)

        if "idiom" in pb_explain:  # I'm assuming TRIPS doesn't have entry that explains idiom
            pass

        else:
            for elem1 in trips[::-1]:  # inner loop TRIPS
                tri_id = str(elem1[0])
                tri_ont = re.sub("ont::", "", tri_id)
                if '-' in tri_ont:
                    tri_ont1 = re.sub("-", "", tri_ont)
                else:
                    tri_ont1 = tri_ont * 1
                tri_synet = elem1[1]
                tri_wnkey = set(elem1[2])
                tri_wn_explain = elem1[3]
                # tri_wn_example = elem1[4]

                score = 1  # starting score

                # step1: Checking if the the trip's entry is exactly in the explanation in propbank
                if tri_ont1 in pb_explain1:
                    score = score * 2

                # step2: Checking if the synsets in trips match the explanation in propbank
                for i in tri_synet:
                    if i in pb_explain0:
                        score = score * 2

                # step3: Checking if the wordnet sense keys in Trips match the wordnet sense keys that verbnet
                # provides for Propbank
                counts = 1
                if len(pb_vn_entry & tri_wnkey) > 0:
                    for i in pb_vn_entry & tri_wnkey:  # find their intersection
                        if word == wn.lemma_from_key(i).name():  # if it has input word key
                            score = score * 2
                        else:  # else count the number of other intersection
                            counts = counts + 1
                score = score * counts

                # step4: checking if there are any combination in the expanation sentence of trips matching the
                # expaination in propbank
                countss = 1
                for explain in tri_wn_explain:
                    length = len(nltk.word_tokenize(explain))
                    for i in range(2, length):  # start from the combination of 2 words
                        mgrams = extract_ngrams(explain, i)  # take the combination
                        for j in mgrams:
                            if j in pb_explain0:
                                countss += 1
                score = score * countss

                # step 5: using Wu Palmer metrics to get the similarity of the word in PropBank and the entry in TRIPS
                s = 0
                for word in pb_explain1:
                    if len(nltk.word_tokenize(word)) == 1:  # only deal with the part with one word
                        word_ontology = word_ont(word)  # find the entry of that word
                        if len(word_ontology) == 0:  # if the word already is one type of ontology
                            similarity = Wu_Palmer_metric(word.upper(), tri_ont.upper())
                        else:
                            for ont_type in word_ontology:
                                ont_type = re.sub("ont::", "", str(ont_type)).upper()
                                similarity = Wu_Palmer_metric(ont_type, tri_ont.upper())
                        if similarity > s:  # find the highest similarity
                            s = similarity
                score = score * s

                candidates.append([elem, elem1, score])

    candidates.sort(key=takesecond, reverse=True)  # sort the candidates with the score from high to low

    return candidates


# Final results
def align(word):
    candidates = candidates_func(word)  # get all the candidates

    l = len(candidates)
    check = []  # using a checking set to check if the entry is already in the final result of not.
    final_result = []
    i = 0
    while i < l:
        if i == 0:  # keep it in the candidates(final result) and append the entry names to the checking set
            check.append(candidates[i][0][0])
            check.append(candidates[i][1][0])
            i += 1
        # else if the candidates has lower score of the previous one and at least one of the entry names is in the
        # checking set or the score is lower than 0.5, delete candidates
        elif candidates[i][2] < candidates[i - 1][2] and (
                candidates[i][0][0] in check or candidates[i][1][0] in check) or candidates[i][2] < 0.5:
            del candidates[i]
            l -= 1

        # else if both entries names are in the checking set, delete it.
        elif candidates[i][0][0] in check and candidates[i][1][0] in check:
            del candidates[i]
            l -= 1
        else:  # else keep it and append the entry names to the checking set
            check.append(candidates[i][0][0])
            check.append(candidates[i][1][0])
            i += 1

    for c in candidates: # append the final result only with the entry names(id) and the score
        final_result.append([c[0][0], c[1][0], c[2]])

    return final_result

###
def main():
    import sys
    with open(sys.argv[1]) as f1:
        inputs = f1.read()
        inputs = list(inputs.split('\n'))

    with open(sys.argv[2], 'r+', ) as f2:
        for i in inputs:
            outputs = align(i)
            for j in outputs:
                    j = str(j)
                    j = j.replace(",", " ")
                    j = j.replace("'", "")
                    j = j.replace("[", "")
                    j = j.replace("]", "")
                    f2.write(j+"\n")
            f2.write("\n\n")
        f2.close()


if __name__ == '__main__':
    main()