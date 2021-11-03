# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import OrderedDict

from MathDataset import MathDataset
from torch.utils.data import DataLoader

from utils import load_vocab, load_raw_math_data

from Tokenizer import Tokenizer

import math
import itertools
from konlpy.tag import Mecab


device = torch.device("cuda:0")
mecab = Mecab()

class MathNet(nn.Module):
    def __init__(self, bert, class_num, seq_len, dmodel):
        super().__init__()
        self.BERT = bert
        self.W = nn.Linear(dmodel * seq_len, class_num)
        self.class_num = class_num
        self.seq_len = seq_len

        for param in self.BERT.parameters():
            param.requires_grad = False

    def forward(self, sequence):
        out = self.BERT(sequence)
        bsize, slen, dmodel = out.shape
        linear_seq = out.reshape(bsize, -1)
        classes = self.W(linear_seq)

        return classes

def main():
    
    solDict = {
            0: "A",    1: "C",    2: "D",    3: "E",    4: "F",
            5: "I",    6: "H",
    }

    tokenizer = Tokenizer()
    vocab = load_vocab("./data/vocab.txt")

    mathNet = torch.load("./pretrained/classifier_base_1100.pth")
    mathNet.eval()

    sentence = "강의 상류에 (가) 지점이, 하류에 (나) 지점이 있고 동시에 두 지점에서 배가 마주 보고 출발하였습니다. 2개의 배는 출발해서 50분 만에 마주쳤고 (가) 지점을 떠난 배는 그 후 30분 만에 (나) 지점에 도착했습니다. 그때 (나) 지점을 떠난 배는 (가) 지점에서 32/5km 떨어진 지점에 있었습니다. 강물은 상류에서 하류로 흐를 때, 두 배가 잔잔한 물에서의 속력이 같을 때 강물의 속력은 몇 km/시 입니까?"

    sym_phr = str(" ".join(mecab.morphs(sentence)))

    idxes, _ = tokenizer.phrase2idxTokens(sym_phr, vocab, 128) 
    pad = 128 - len(idxes)
    pad = pad * [0]
    idxes.extend(pad)

    idxesQuestions = torch.tensor([idxes])
    with torch.no_grad():
        mathClass = mathNet(idxesQuestions.to(device))
        mathClass = int(torch.argmax(mathClass).detach().cpu().numpy())

    classname = solDict[mathClass]
    
    print("classname : ", classname)


if __name__ == '__main__':
    main()