from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import yaml
from datetime import datetime
import copy
from logging import getLogger, StreamHandler, INFO, DEBUG
from collections import Counter
import sys
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, vocab_size,embed_size,hidden_size,batch_size,lstm_layers,dropout):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        "encoder part"
        self.hidden = 0
        self.encoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=PAD_TAG[1])
        self.encoderlstm = nn.LSTM(embed_size,self.hidden_size,num_layers=self.lstm_layers,dropout=dropout,bidirectional=False,batch_first=True)
    def encode_init_hidden(self,size):
        return (torch.randn(self.lstm_layers,size,self.hidden_size,device=device),torch.randn(self.lstm_layers,size,self.hidden_size,device=device))
    def forward(self,sentences,mask):
        self.hidden = self.encode_init_hidden(sentences.size(0))
        embedded = self.encoderembedding(sentences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,mask[1],batch_first = True)
        output,self.hidden =self.encoderlstm(packed,self.hidden)
        output,legnth = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first =True)
        return output
class Decoder(nn.Module):
    def __init__(self, vocab_size,embed_size,hidden_size,batch_size,lstm_layers,dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        "attentionpart"
        self.attn = attn()
        "decoder part"
        self.decoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=PAD_TAG[1])
        self.deoderlstm = nn.LSTM(embed_size,self.hidden_size,num_layers=self.lstm_layers,dropout=dropout,bidirectional=False,batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(1)
    def forward(self,words,enoder_out,t,mask):
        embedded = self.decoderembedding(words)
        lstm_output,self.hidden = self.deoderlstm(embedded,self.hidden)
        output = self.out(lstm_output).squeeze(1)
        output = self.softmax(output)
        return output
def splitmecab(sentence):
    tokenlist=[]
    neologd_dic_path = "/data/home/katsumi/mecab/dic/mecab-ipadic-neologd" 
    tokenizer = mecab_tokenizer.Tokenizer("-Ochasen -d " + neologd_dic_path)
    token_list = tokenizer.tokenize(sentence)
    for token in token_list:
        tokenlist.append(token.surface)
    return tokenlist
def get_train_data(TRAIN__TOKEN_FILE, max_vocab_size):
    logger.info("========= START TO GET TOKEN ==========")
    with open(file=TRAIN_TOKEN_FILE[0], encoding="utf-8") as question_text_file, open(file=TRAIN_TOKEN_FILE[1],
                                                                                    encoding="utf-8") as answer_text_file:
        answer_lines = answer_text_file.readlines()
        question_lines = question_text_file.readlines()
    vocab = []
    for line1, line2 in zip(answer_lines, question_lines):
        vocab.extend(line1.replace("\t", " ").split())
        vocab.extend(line2.replace("\t", " ").split())
    logger.debug(vocab)
    vocab_counter = Counter(vocab)
    vocab = [v[0] for v in vocab_counter.most_common(max_vocab_size)]
    word_to_id = {v: i + 4 for i, v in enumerate(vocab)}
    word_to_id[UNKNOWN_TAG[0]] = UNKNOWN_TAG[1]
    word_to_id[EOS_TAG[0]] = EOS_TAG[1]
    word_to_id[SOS_TAG[0]] = SOS_TAG[1]
    word_to_id[PAD_TAG[0]] = PAD_TAG[1]
    id_to_word = {i: v for v, i in word_to_id.items()}
    logger.debug(id_to_word)
    train_data = []
    for question_line, answer_line in zip(question_lines, answer_lines):
        if len(question_line) < 1 or len(answer_line) < 1:
            continue
        input_words = [word_to_id[word] if word in word_to_id.keys() else word_to_id[UNKNOWN_TAG[0]] for word in
                       question_line.split()]
        output_words = [word_to_id[word] if word in word_to_id.keys() else word_to_id[UNKNOWN_TAG[0]] for word in
                        answer_line.split()]
        output_words.append(EOS_TAG[1])
        if len(input_words) > 0 and len(output_words) >0:
            train_data.append([input_words, output_words])
    return train_data, word_to_id, id_to_word
def tokenize():
    logger.info("========= START TO TOKENIZE ==========")
    tokenizer = mecab_tokenizer.Tokenizer("-Ochasen -d " + neologd_dic_path)
    train_tokenized = []
    with open(TRAIN_FILE[0],encoding="utf-8") as f1, open(TRAIN_FILE[1], encoding="utf-8") as f2:
        for line1, line2 in zip(f1.readlines(), f2.readlines()):
            if len(line1) < 1 or len(line2) < 1:
                continue
            try:
                train_tokenized.append((" ".join([token.surface for token in tokenizer.tokenize(line1)]),
                                            " ".join([token.surface for token in tokenizer.tokenize(line2)])))
            except: 
                pass
    with open(TRAIN_TOKEN_FILE[0], "wt", encoding="utf-8") as f1, open(TRAIN_TOKEN_FILE[1], "wt",
                                                                                  encoding="utf-8") as f2:
        for line1, line2 in train_tokenized:
            f1.write(line1 + "\r\n")
            f2.write(line2 + "\r\n")
def makeminibatch(training_data):
    n = len(training_data)
    mini_batch_size = int(n/batch_size)
    random.shuffle(training_data)
    batch_training_data = []
    for i in range(0,n,mini_batch_size):
        if i+batch_size>n:
            batch_training_data.append(training_data[i:])
        else:
            batch_training_data.append(training_data[i:i+batch_size])
    return batch_training_data
def padding_mask(sentences,word_to_id):
    mask = sentences.data.eq(word_to_id[PAD_TAG[0]])
    return (mask, sentences.size(1) - mask.sum(1))
def train(train_data, word_to_id, id_to_word, model_path):
    encoder = Encoder(len(word_to_id),embed_size,hidden_size,batch_size,lstm_layers,dropout).to(device)
    decoder = Decoder(len(word_to_id),embed_size,hidden_size,batch_size,lstm_layers,dropout).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01,weight_decay=1e-4)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01,weight_decay=1e-4)
    logger.info("========= WORD_SIZE={} ==========".format(len(word_to_id)))
    logger.info("========= TRAIN_SIZE={} =========".format(len(train_data)))
    logger.info("========= START_TRAIN ==========")
    all_EPOCH_LOSS = []
    for epoch in range(epoch_num):
        total_loss = 0
        logger.info("=============== EPOCH {} {} ===============".format(epoch + 1, datetime.now()))
        batch_training_data = makeminibatch(train_data)
        for count,batch_data in enumerate(batch_training_data):
            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            logger.debug("===== {} / {} =====".format(count, len(batch_training_data)))
            batch_data.sort(key=lambda batch_data:len(batch_data[0]),reverse=True)
            inputsentences = [data[0] for data in batch_data]
            outputsentences = [data[1] for data in batch_data]
            input_max_len = max([len(seq) for seq in inputsentences])
            output_max_len = max([len(seq) for seq in outputsentences])
            inputsentences = [sentence+[word_to_id[PAD_TAG[0]]]*(input_max_len-len(sentence)) for sentence in inputsentences]
            outputsentences = [sentence+[word_to_id[PAD_TAG[0]]]*(output_max_len-len(sentence)) for sentence in outputsentences]
            inputsentences = torch.tensor(inputsentences,dtype=torch.long,device=device)
            outputsentences = torch.tensor(outputsentences,dtype=torch.long,device=device)
            mask = padding_mask(inputsentences,word_to_id)
            encoder_out = encoder(inputsentences,mask)
            decoder_input = torch.tensor([SOS_TAG[1]] * len(inputsentences)).unsqueeze(1)
            decoder.hidden = encoder.hidden
            for t in range(outputsentences.size(1)):
                decoder_output = decoder(decoder_input,encoder_out,t,mask)
                loss += F.nll_loss(decoder_output, outputsentences[:, t], ignore_index = PAD_TAG[1], reduction = "sum")
                decoder_input = outputsentences[:, t].unsqueeze(1) #teacher forcing
                #decoder_input = decoder_output.squeeze(1).argmax(1,keepdim=True) #else
                break
            break
            loss /= outputsentences.data.gt(0).sum().float()
            sys.exit()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss
            logger.info("=============== loss: %s ===============" % loss)
        total_loss = total_loss/len(batch_training_data)
        logger.info("=============== total_loss: %s ===============" % total_loss)
        all_EPOCH_LOSS.append(total_loss)
    [logger.info("================ batchnumber: {}---loss: {}=======================".format(batchnumber,loss)) for batchnumber,loss in enumerate(all_EPOCH_LOSS)]
    torch.save(encoderdecoder.state_dict(), model_path)
def word_to_id(sentence,word_to_id):
    word_ids = [word_to_id.get(token, UNKNOWN_TAG[1]) for token in sentence]
"""
config
"""
config = yaml.load(open("config.yml", encoding="utf-8"))
TRAIN_FILE = (config["train_file"]["q"], config["train_file"]["a"])
MODEL_FILE = config["encdec"]["model"]
TRAIN_TOKEN_FILE = (config["train_token_file"]["q"],config["train_token_file"]["a"])
epoch_num = int(config["encdec"]["epoch"])
batch_size = int(config["encdec"]["batch"])
embed_size = int(config["encdec"]["embed"])
hidden_size = int(config["encdec"]["hidden"])
dropout = float(config["encdec"]["dropout"])
lstm_layers = int(config["encdec"]["lstm_layers"])
max_vocab_size = int(config["encdec"]["max_vocab_size"])
neologd_dic_path = config["neologd_dic_path"]
save_model_path = config["save_model_path"]
PAD_TAG = ("<PAD>",0)
UNKNOWN_TAG = ("<UNK>", 1)
EOS_TAG = ("<EOS>", 3)
SOS_TAG = ("<SOS>",2)
def main():
    #tokenize()
    traindata,word_to_id,id_to_word =  get_train_data(TRAIN_TOKEN_FILE,max_vocab_size)
    train(traindata,word_to_id,id_to_word,MODEL_FILE)
    
if  __name__ == '__main__':
    print(torch.cuda.is_available())
    main()