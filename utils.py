import numpy as np
import re
import random as rd

def load_vocab(PATH):
    vocab = []
    with open(PATH, 'r', encoding="utf-8-sig") as f:
        raw = f.readlines()
        for line in raw:
            vocab.append(line.strip())

    return vocab

def load_raw_math_data(PATH):
    questions = []
    labels = []
    with open(PATH, 'r', encoding="utf-8-sig") as f:
        raw = f.readlines()
        for line in raw:
            data = line.strip().split('\t')
            question = data[0]
            if len(data) > 1:
                label = int(data[1])
                labels.append(label)
            questions.append(question)
    return questions, labels


def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [1] * nclasses                                                      
    for item in labels:                                                         
        count[item] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight 

