import torch
import torch.nn as nn
import torch.optim as optim

from bert_pytorch import BERT

from utils import load_raw_math_data, load_vocab, make_weights_for_balanced_classes

from adabelief_pytorch import AdaBelief

from MathDataset import MathDataset

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import random as rd
import os

import numpy as np

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

if __name__ == "__main__":
    
    if not os.path.exists("./output"):
        os.makedirs("./output")

    questions, labels = load_raw_math_data("./data/traindata.tsv")
    vocab = load_vocab("./data/vocab.txt")
    seq_len = 128
    number_of_classes = 7
    data_len = len(questions)

    device = torch.device("cuda:0")
    bert = torch.load("./pretrained/bert_trained.model.ep1000")

    math_dataset = MathDataset(questions, labels, vocab)

    weights = make_weights_for_balanced_classes(labels, number_of_classes)                   
    weights = torch.DoubleTensor(weights)      
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                                           
    train_dataset = torch.utils.data.DataLoader(math_dataset, batch_size = 128, sampler = sampler, pin_memory=True)  
    model = MathNet(bert, number_of_classes, 128, 468).to(device)
    optimizer = AdaBelief(model.parameters(), lr = 0.0001, eps=1e-8, betas=(0.9, 0.999), weight_decouple = True, rectify = False)

    #optimizer = optim.SGD(model.parameters(), lr = 0.0001, weight_decay = 0.5)

    criterionA = nn.CrossEntropyLoss()
    
    pbar = tqdm(range(0,300000))

    for epoch in pbar:

        avg_loss = 0.0

        model.train()
        # Train
        for batch in train_dataset:
            questions = batch[0].to(device)
            labels = batch[2].to(device)

            pred_class = model(questions)

            loss = criterionA(pred_class, labels)

            optimizer.zero_grad()
        
            loss.backward()
        
            optimizer.step()

            avg_loss += loss.item()

        if epoch and epoch % 20 == 0:
            print("model save...")
            torch.save(model, "./output/classifier_base"+f"_{epoch}"+".pth")

        cur_epoch = avg_loss / len(train_dataset)
        pbar.set_description("train loss : %f " % cur_epoch)
