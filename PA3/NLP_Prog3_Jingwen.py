#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Jingwen
"""

##
import jsontrips

ontology_dict = jsontrips.ontology()


# recursive function to get all the parents(Changing from PA2)
def parent_recursion(x, parents):
    try:
        y = ontology_dict[x]['parent']
        parents.append(y)
        y, parents = parent_recursion(y, parents)
        return parents
    except:
        return parents


##
# get the similarity (Changing from PA2)
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


##
# Changing the input format
import shlex
import re
import nltk


def change_format(file):
    Story = []
    for line in file:
        line = line.replace(' (THE-SET', ')\n(THE-SET')  # for input example 6
        lines = line.split('\n')
        new_line = []
        for l in lines:
            if l.startswith('('):
                l = shlex.split(l)
                for i in range(len(l)):
                    l[i] = l[i].replace('(', '')
                    l[i] = l[i].replace(')', '')
                    if i == 3:
                        l[i] = re.sub(r"\W+|_", " ", l[i])
                        l[i] = nltk.word_tokenize(l[i])
                        for j in range(len(l[i])):
                            l[i] = l[i][0]
                    else:
                        l[i] = [re.sub(r"ONT::", "", l[i])]  # delete ONT::
                        for j in range(len(l[i])):
                            l[i] = l[i][0].upper()
                new_line.append(l)
        Story.append(new_line)
    return Story


## Get Candidate:
import numpy as np


def get_candidate(Story):
    candidates = []
    # case a: join all the sentences together, for case that are not related to the nearest previous one later
    a = [x for j in Story for x in j]
    count = 0

    for i in range(len(Story)):  # make the format
        candidates.append([])

    # start:
    for i in range(len(Story)):
        if i == 0:  # first case which will look like ()
            candidates[i].append(())
        else:
            for line1 in Story[i]:
                count += 1  # for case a

                if "SELF" in line1[1] or "SELVES" in line1[1]:  # for input example 4, the second sentence
                    for LINE1 in Story[i]:
                        if line1[3][0] == LINE1[3][0] and line1[3] != LINE1[3]:
                            # get the candidate and assign similarity to 2
                            candidates[i].append(('COREF', line1[-1], LINE1[-1], 2))

                elif "THE-SET" in line1[0]:  # for input example 6
                    for LINE1 in Story[0]:
                        if line1[1] == LINE1[1]:
                            # get the candidate and assign similarity to 2
                            candidates[i].append(('COREF', line1[-1], LINE1[-1], 2))

                else:
                    # using similarity for the ontologies to get the candidate
                    Wu_similarity = 0
                    for line2 in Story[i - 1]:
                        # situation 1: the <word> and <type> are the same, assign similarity to 2
                        if line1[1:-1] == line2[1:-1] and ('A' not in line1):
                            candidates[i].append(('COREF', line1[-1], line2[-1], 2))

                        elif ('PRO' in line1) or ('THE' in line1):  # situation 2: Pronoun
                            # Using wu palmer to get all the other candidates
                            s = Wu_Palmer_metric(line1[2], line2[2])
                            if s > Wu_similarity or s > 0.95:
                                Wu_similarity = s
                                if Wu_similarity == 1 and line1[1] != line2[1]:
                                    Wu_similarity = Wu_similarity
                                # get the candidate and assign similarity = Wu_similarity
                                candidate1 = ('COREF', line1[-1], line2[-1], Wu_similarity)
                                candidates[i].append(candidate1)
                                candidate1 = ()

                        elif ('PRO-SET' in line1) and ("SET" in Story[0][0][0]):  # situation 3: PRO-SET
                            # Using wu palmer to get all the other candidates
                            s = Wu_Palmer_metric(line1[2], line2[2])
                            if s > Wu_similarity or s > 0.95:
                                Wu_similarity = s
                                if Wu_similarity == 1 and line1[1] != line2[1]:
                                    # get the candidate and assign similarity = Wu_similarity
                                    Wu_similarity = Wu_similarity
                                candidate1 = ('COREF', line1[-1], line2[-1], Wu_similarity)
                                candidates[i].append(candidate1)
                                candidate1 = ()

                    # for input example 1 and input example 6 where refer to the one that not from story(i-1)
                    if i >= 2 and ('A' not in line1): #only i>2 need to be compared
                        b = []
                        for n in reversed(range(count - 2)):  # reversing check
                            if line1[:-1] == a[n][:-1]:
                                b.append(a[n][-1])
                        if len(b) > 0:
                            candidates[i].append(('COREF', line1[-1], b[0], 0.9))
    return candidates


##
# For outputs:
def final_outputs(candidates):
    outputs = []

    for i in range(len(candidates)):  # getting the format
        outputs.append([])

    for i in range(len(candidates)):  # basic case
        if i == 0:
            outputs[i] = candidates[i]  # first case will be like ()
        else:
            for c in candidates[i]:
                if c[1][0] == c[2][0] and c[3] == 2:  # keep the candidates that are 100% matching
                    outputs[i].append(c[:-1])

    # update candidates: reflexive constraints
    # delete extra candidates
    for i in range(1, len(candidates)):
        if len(outputs[i]) == 0:
            a = 0
        else:
            a = len(candidates[i])
        while a > 1:
            for c in candidates[i]:
                for o in outputs[i]:
                    if c[1] == o[1] or c[2] == o[2]:
                        candidates[i].remove(c)
                        a = len(candidates[i])

    # update candidates: compare the similarity
    # delete extra candidates
    for i in range(1, len(candidates)):
        for c in candidates[i]:
            # choose the candidate that has higher similarity
            for c2 in candidates[i]:
                if c[1] == c2[1] and c[2] != c2[2] and c2[3] > c[3]:
                    candidates[i].remove(c)
                elif c[1] == c2[1] and c[2] != c2[2] and c2[3] < c[3]:
                    candidates[i].remove(c2)

    # final out put:
    for i in range(1, len(candidates)):
        for c in candidates[i]:
            if c[:-1] not in outputs[i]:
                outputs[i].append(c[:-1])

    return outputs



##
def main():
    import sys
    with open(sys.argv[1]) as f1:
        inputs = f1.read()
        inputs = inputs.split('\n\n')

    with open(sys.argv[2], 'r+', ) as f2:
        Story = change_format(inputs)
        candidates = get_candidate(Story)
        outputs = final_outputs(candidates)

        for line in outputs:
            for j in line:
                j = str(j)
                j = j.replace("'", "")
                j = j.replace(",", "")
                f2.write(j + ' ')
            f2.write("\n")
        f2.close()


if __name__ == '__main__':
    main()
