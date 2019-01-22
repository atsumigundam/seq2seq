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
import MeCab
import mecab_tokenizer
import shape_commentator
import copy
from logging import getLogger, StreamHandler, INFO, DEBUG
from collections import Counter
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
#madaattentiontuketenai
#shape_commentator.comment(In[len(In)-2],globals(),locals())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size,batch_size,lstm_layers,dropout):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.decode_max_size = 20
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        "encoder part"
        self.encoderhidden = 0
        self.encoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=3)
        self.encoderlstm = nn.LSTM(embed_size,self.hidden_size//2,num_layers=self.lstm_layers,dropout=dropout,bidirectional=True,batch_first=True)
        "decoder part"
        self.decoderhidden = 0
        self.decoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=3)
        self.decoderlstm = nn.LSTM(embed_size,self.hidden_size//2,num_layers=self.lstm_layers,dropout=dropout,bidirectional=True,batch_first=True)
        self.decoderout = nn.Linear(self.hidden_size,self.vocab_size)
        "attention part"
        #attentionnobaainihadecoderhatokengotonisuru
    def encode(self, sentences,input_lengths,train):
        if train == True:
            self.encoderhidden = self.encode_init_hidden(sentences.size(0))
            embedded = self.encoderembedding(sentences)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,batch_first = True)
            output,self.encoderhidden =self.encoderlstm(packed,self.encoderhidden)
            output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first =True)
        elif train == False:
            sentences = sentences.view(-1,len(sentences))
            self.encoderhidden = self.encode_init_hidden(sentences.size(0))
            embedded = self.encoderembedding(sentences)
            output,self.encoderhidden =self.encoderlstm(embedded,self.encoderhidden)
        return output,self.encoderhidden
    def decode(self,sentences,hiddens,train):
        if train == True:
            self.decoderhidden = self.decode_init_hidden(len(sentences),hiddens)
            embedded = self.decoderembedding(sentences)
            output,self.decoderhidden =self.decoderlstm(embedded,self.decoderhidden)
            output = self.decoderout(output)
        return output
    def predictdecode(self,hiddens):
        max_length = self.decode_max_size
        self.decoderhidden = self.decode_init_hidden(1,hiddens)
        wid = 2
        result = []
        for i in range(max_length):
            wid = torch.tensor(wid,dtype=torch.long,device = device).view(1,-1)
            embedded =self.decoderembedding(wid)
            output,self.decoderhidden =self.decoderlstm(embedded,self.decoderhidden)
            output = self.decoderout(output)
            output = F.softmax(output[0][0],dim =0)
            print(output)
            wid = output.argmax()
            print(wid)
            if wid ==1:
                break
            result.append(int(wid))
        return result
    def forward(self,inputsentences,inputpaddingnumberlist,outputsentences,outputpaddingnumberlist):
        inputsentences = torch.tensor(inputsentences,dtype=torch.long,device=device)
        train_outputsentences = torch.tensor(outputsentences,dtype=torch.long,device=device)
        value_outputsentences = copy.deepcopy(outputsentences)
        value_outputsentences =insertEOS(value_outputsentences,outputpaddingnumberlist)
        value_outputsentences = torch.tensor(value_outputsentences,dtype=torch.long,device=device)
        sos = torch.tensor(SOS_TAG[1],dtype=torch.long,device=device).view(1,-1).expand(len(outputsentences),1)
        train_outputsentences=torch.cat([sos,train_outputsentences],dim = 1)
        encoder_output,encoder_hidden_taple =self.encode(inputsentences,inputpaddingnumberlist,True)
        decoderoutput =self.decode(train_outputsentences,encoder_hidden_taple,True)
        critetion = nn.CrossEntropyLoss(ignore_index=3)
        batchloss = 0
        for(decodeoutput,padding_number,targetsentence) in zip(decoderoutput,outputpaddingnumberlist,value_outputsentences):
            decodeoutput = decodeoutput[:padding_number+1]
            targetsentence = targetsentence[:padding_number+1]
            loss = critetion(decodeoutput,targetsentence)
            batchloss = batchloss+loss
        batchloss = batchloss/len(inputsentences)
        return batchloss
    def predict(self,inputsentence):
        inputsentences = torch.tensor(inputsentence,dtype=torch.long,device=device)
        print(inputsentences)
        encoder_output,encoder_hidden_taple =self.encode(inputsentences,1,False)
        decoderoutput =self.predictdecode(encoder_hidden_taple)
        return decoderoutput
    def encode_init_hidden(self,size):
        return (torch.randn(self.lstm_layers*2,size,self.hidden_size//2,device=device),torch.randn(self.lstm_layers*2,size,self.hidden_size//2,device=device))
    def decode_init_hidden(self,size,encode_hiddens):
        return encode_hiddens
def insertEOS(sentences,numbers):
    for (sentence,number) in zip(sentences,numbers):
        if len(sentence) == number:
            sentence.append(EOS_TAG[1])
        else:
            sentence.insert(number,EOS_TAG[1])
    return sentences
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
def padding(maxlen,sentence,word_to_id):
    sentence.extend([word_to_id[PAD_TAG[0]] for i in range(maxlen - len(sentence))])
    return sentence
def train(train_data, word_to_id, id_to_word, model_path):
    encoderdecoder = EncoderDecoder(len(word_to_id),embed_size,hidden_size,batch_size,lstm_layers,dropout).to("cuda")
    optimizer = optim.Adam(encoderdecoder.parameters(), lr=0.01,weight_decay=1e-4)
    logger.info("========= WORD_SIZE={} ==========".format(len(word_to_id)))
    logger.info("========= TRAIN_SIZE={} =========".format(len(train_data)))
    logger.info("========= START_TRAIN ==========")
    all_EPOCH_LOSS = []
    for epoch in range(epoch_num):
        total_loss = 0
        logger.info("=============== EPOCH {} {} ===============".format(epoch + 1, datetime.now()))
        batch_training_data = makeminibatch(train_data)
        for count,batch_data in enumerate(batch_training_data):
            logger.debug("===== {} / {} =====".format(count, len(batch_training_data)))
            batch_data.sort(key=lambda batch_data:len(batch_data[0]),reverse=True)
            inputsentences = [data[0] for data in batch_data]
            outputsentences = [data[1] for data in batch_data]
            inputlen_sen = [len(seq) for seq in inputsentences]
            outputlen_sen = [len(seq) for seq in outputsentences]
            inputsentences = [sentence if len(sentence)==max(inputlen_sen) else sentence+[word_to_id[PAD_TAG[0]] for i in range(max(inputlen_sen) - len(sentence))] for sentence in inputsentences]
            input_padding_list = [sentence.index(word_to_id[PAD_TAG[0]]) if word_to_id[PAD_TAG[0]] in sentence else len(sentence) for sentence in inputsentences]
            outputsentences = [sentence if len(sentence)==max(outputlen_sen) else sentence+[word_to_id[PAD_TAG[0]] for i in range(max(outputlen_sen) - len(sentence))] for sentence in outputsentences]
            output_padding_list = [sentence.index(word_to_id[PAD_TAG[0]]) if word_to_id[PAD_TAG[0]] in sentence else len(sentence) for sentence in outputsentences]
            encoderdecoder.zero_grad()
            loss = encoderdecoder(inputsentences,input_padding_list,outputsentences,output_padding_list)
            loss.backward()
            optimizer.step()
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
UNKNOWN_TAG = ("<UNK>", 0)
EOS_TAG = ("<EOS>", 1)
SOS_TAG = ("<SOS>",2)
PAD_TAG = ("<PAD>",3)
def main():
    #tokenize()
    traindata,word_to_id,id_to_word =  get_train_data(TRAIN_TOKEN_FILE,max_vocab_size)
    train(traindata,word_to_id,id_to_word,MODEL_FILE)
    print("predict")
    the_model = EncoderDecoder(len(word_to_id),embed_size,hidden_size,batch_size,lstm_layers,dropout).to("cuda")
    the_model.load_state_dict(torch.load(MODEL_FILE))
    with torch.no_grad():
        while True:
            input_ = input("INPUT>>>")
            sentence = input_
            word_ids = [word_to_id.get(token, UNKNOWN_TAG[1]) for token in splitmecab(sentence)]
            idnolist=the_model.predict(word_ids)
            answer = [id_to_word.get(bocabnum, UNKNOWN_TAG[0]) for bocabnum in idnolist]
            print(answer)
if  __name__ == '__main__':
    print(torch.cuda.is_available())
    main()