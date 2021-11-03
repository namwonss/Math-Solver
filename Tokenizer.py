from konlpy.tag import Mecab
#from Symbolizer import Symbolizer
from utils import load_raw_math_data
import random as rd

class Tokenizer:
    def __init__(self):
        self.mecab = Mecab()
        #self.symbolizer = Symbolizer()

    def whitespace_tokenize(self, data):
        data = data.strip()   
        if not data:
            return []
        tokens = data.split()  
        return tokens

    def tokenize(self, phrase , forVocab = False):
        output_tokens = []
        if forVocab:
            output_tokens = ['PAD', 'UNK', 'MASK', 'BOS', 'EOS']

        temp_phrase = phrase

        for wst in self.whitespace_tokenize(temp_phrase):
            count = 0
            for token, pos in self.mecab.pos(wst):
                tk = token

                if count > 0:
                    tk = "##" + tk
                    if forVocab:
                        if tk in output_tokens:
                            continue
                        output_tokens.append(tk)
                else:
                    count += 1
                    if forVocab:
                        if tk in output_tokens:
                            continue
                        output_tokens.append(tk)

                if not forVocab:
                    output_tokens.append(tk)

        return output_tokens

        # END OF ERA
    def phrase2idxTokens(self, symbolized, vocab, seq_len, isBert = False):
        symbolized = "BOS "+ symbolized + "EOS"
        tokens = self.tokenize(symbolized)
        bert_output = []
        idxes = []

        for token in tokens:
            if token in vocab:
                idxes.append(vocab.index(token))

        if isBert:
            idxes, bert_output = self.masking_tokens(idxes)

        return idxes, bert_output

    def masking_tokens(self, tokens):
        number_of_mask = int(len(tokens) * 0.15)
        copied_tokens = tokens.copy()

        if not number_of_mask:
            number_of_mask = 1
        
        samples = rd.sample(range(1,len(tokens) -1), number_of_mask)

        toss = rd.random()

        for sample in samples:
            if toss > 0.2:
                copied_tokens[sample] = 2
            elif toss > 0.1 and toss <= 0.2:
                copied_tokens[sample] = rd.randint(10,1000)

        bert_output = [tokens[sample] for sample in samples]

        return copied_tokens, bert_output

if __name__=="__main__":
    
    #symbolizer = Symbolizer()
    tokenizer = Tokenizer()
    symbolized_corpus = ""

    questions, _ = load_raw_math_data("./data/corpus.tsv")
    mecab = Mecab()
    # MAKE SYMOBLIZED CORPUS
    for question in questions:
        sym_phr = str(" ".join(mecab.morphs(question)))

        symbolized_corpus += sym_phr
    
    vocab = tokenizer.tokenize(symbolized_corpus, forVocab=True)

    with open('./data/vocab.txt', 'w', encoding='utf-8-sig') as f:
        for v in vocab:
            f.write(v + '\n')
