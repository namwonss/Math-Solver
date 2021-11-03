from torch.utils.data import Dataset, DataLoader
import tqdm
import torch
import random

# User Lib
from utils import load_raw_math_data, load_vocab

#from Symbolizer import Symbolizer
from Tokenizer import Tokenizer

from konlpy.tag import Mecab

# BERT MAX SEQ = 128

class MathDataset(Dataset):
    def __init__(self, questions, labels, vocab, seq_len = 128, isBert = False):
        self.vocab = vocab
        self.questions = questions
        self.seq_len = seq_len
        self.labels = labels
        self.tokenizer = Tokenizer()
        #self.symbolizer = Symbolizer()
        self.isBert = isBert
        self.mecab = Mecab()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        cur_phrase = self.questions[index]
        idx_label = []
        #symbolized, _ = self.symbolizer.parsing(cur_phrase)
        # symbolized, _ = preprocessing.parsing(cur_phrase)
        symbolized = " ".join(self.mecab.morphs(cur_phrase))

        idxes, bert_output = self.tokenizer.phrase2idxTokens(symbolized, self.vocab, self.seq_len, self.isBert)

        if len(idxes) > self.seq_len:
            idxes = idxes[:self.seq_len-1]
            idxes.append(4)

        bert_input_pad_num = self.seq_len - len(idxes)
        bert_output_pad_num = self.seq_len - len(bert_output)

        input_pad_list = [0] * bert_input_pad_num
        output_pad_list = [0] * bert_output_pad_num

        if not self.isBert:
            idx_label = torch.tensor(self.labels[index], dtype=torch.int64)

        idxes.extend(input_pad_list)
        bert_output.extend(output_pad_list)

        return torch.tensor(idxes, dtype=torch.int64), torch.tensor(bert_output, dtype=torch.int64), idx_label
