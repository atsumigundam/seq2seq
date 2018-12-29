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
import MeCab
import mecab_tokenizer
import shape_commentator
from pyknp import Juman
import copy
#madaattentiontuketenai
#shape_commentator.comment(In[len(In)-2],globals(),locals())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderDecoderbypart(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size,batch_size):
        super(EncoderDecoderbypart, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.decode_max_size = 20
        self.batch_size = batch_size
        "encoder part"
        self.encoderhidden = 0
        self.encoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.encoderlstm = nn.LSTM(embed_size,hidden_size,num_layers=1,batch_first=True)
        "decoder part"
        self.decoderhidden = 0
        self.decoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.decoderlstm = nn.LSTM(embed_size,hidden_size,num_layers=1,batch_first=True)
        self.decoderout = nn.Linear(hidden_size,self.vocab_size)
class EncoderDecoderbytoken(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size,batch_size):
        super(EncoderDecoderbytoken, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.decode_max_size = 20
        self.batch_size = batch_size
        "encoder part"
        self.encoderhidden = 0
        self.encoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.encoderlstm = nn.LSTM(embed_size,hidden_size,num_layers=1,batch_first=True)
        "decoder part"
        self.decoderhidden = 0
        self.decoderembedding = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        self.decoderlstm = nn.LSTM(embed_size,hidden_size,num_layers=1,batch_first=True)
        self.decoderout = nn.Linear(hidden_size,self.vocab_size)
    def encode(self, sentences,input_lengths,train):
        if train == True:
            self.encoderhidden = self.encode_init_hidden(sentences.size(0))
            embedded = self.encoderembedding(sentences)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,batch_first = True)
            output,self.encoderhidden =self.encoderlstm(packed)
            output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first =True)
        elif train == False:
            embedded = self.encoderembedding(sentences)
            self.encoderhidden = self.encode_init_hidden(1)
            output,self.encoderhidden =self.encoderlstm(embedded.view(1, len(sentences),-1),self.encoderhidden)
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
        wid = 1
        result = []
        for i in range(max_length):
            embedded =self.decoderembedding(torch.tensor(wid,dtype=torch.long,device = device))
            output,self.decoderhidden =self.decoderlstm(embedded.view(1, 1,-1),self.decoderhidden)
            output = self.decoderout(output)
            output = F.softmax(output[0][0],dim =0)
            wid = output.argmax()
            if wid == 2:
                break
            result.append(int(wid))
        print(result)
        return result
    def forward(self,inputsentences,inputpaddingnumberlist,inputlength,outputsentences,outputpaddingnumberlist):
        inputsentences = torch.tensor(inputsentences,dtype=torch.long,device=device)
        inputlength = torch.tensor(inputlength,dtype=torch.long,device=device)
        print(inputsentences.size())
        train_outputsentences = copy.deepcopy(outputsentences)
        value_outputsentence = copy.deepcopy(outputsentences)
        loss_outputsentence = []
        for sentence in train_outputsentences:
            sentence.insert(0,1)
        for (out_sentence,paddingnumber) in zip(value_outputsentence,outputpaddingnumberlist):
            if paddingnumber!=-1:
                out_sentence= [i for i in out_sentence if not i == 0]
                out_sentence.append(2)
                loss_outputsentence.append(out_sentence)
            else:
                out_sentence.append(2)
                loss_outputsentence.append(out_sentence)
        train_outputsentences = torch.tensor(train_outputsentences,dtype=torch.long,device=device)
        encoder_output,encoder_hidden_taple =self.encode(inputsentences,inputlength,True)
        print(encoder_output.size())
        print("---------")
        decoderoutput =self.decode(train_outputsentences,encoder_hidden_taple,True)
        critetion = nn.CrossEntropyLoss()
        loss = 0
        for(decode,padding_number,targetsentence) in zip(decoderoutput,outputpaddingnumberlist,loss_outputsentence):
            if padding_number!=-1:
                decode=decode[:padding_number+1]
                targetsentence = torch.tensor(targetsentence,dtype=torch.long,device=device)
                loss += critetion(decode,targetsentence)
            else:
                targetsentence = torch.tensor(targetsentence,dtype=torch.long,device=device)
                loss += critetion(decode,targetsentence)
        loss = loss/len(inputsentences)
        return loss
    def predict(self,inputsentence):
        inputsentences = torch.tensor(inputsentence,dtype=torch.long,device=device)
        encoder_output,encoder_hidden_taple =self.encode(inputsentences,1,False)
        decoderoutput =self.predictdecode(encoder_hidden_taple)
        return decoderoutput
    def encode_init_hidden(self,size):
        return (torch.randn(1,size,self.hidden_size,device=device),torch.randn(1,size,self.hidden_size,device=device))
    def decode_init_hidden(self,size,encode_hiddens):
        #hiddens = (torch.randn(1,size,self.hidden_size))
        #for i,encode_hidden in enumerate(encode_hiddens):
            #hiddens[0][i] = encode_hidden
        #print(encode_hiddens)
        #print(hiddens)
        return encode_hiddens
def minidataload(inputpath,outputpath):
    inputsentencelist = []
    outputsentencelist = []
    convsentencelist = []
    with open(inputpath) as f:
        for s_line in f:
            inputsentencelist.append(s_line.strip())
    with open(outputpath) as f:
        for s_line in f:
            outputsentencelist.append(s_line.strip())
    #convsentencelist.append(inputsentencelist)
    #convsentencelist.append(outputsentencelist)
    #print(convsentencelist[0][0])#初めまして。
    #print(convsentencelist[1][0])#初めまして。よろしくお願いします
    return inputsentencelist,outputsentencelist
def loadmeidaicorpus(meidaipath):
    with open(meidaipath) as f:
        for s_line in f:
            input,output = s_line.strip().split("output: ")
            print(input.replace("input: ",""))
            print(output)
            print("-----------")
def jumancheck(sentence):
    jumanpp = Juman()
    result = jumanpp.analysis(sentence)
    hinshilist = []
    tokenlist = []
    for mrph in result.mrph_list():
        tokenlist.append(mrph.midasi)
        if mrph.hinsi!="名詞":
            hinshilist.append(mrph.hinsi)
        else:
            if "カテゴリ" in mrph.imis:
                #categorylist = (mrph.imis.split(" ")
                #print(mrph.imis.split(" "))
                #hinshilist.append(mrph.imis.split("カテゴリ:")[1])
                print(mrph.imis.split("カテゴリ:")[1])
            else:
                hinshilist.append("NIL")
    return hinshilist,tokenlist
def splitmecab(sentence):
    tokenlist=[]
    hinshilist = []
    neologd_dic_path = "/data/home/katsumi/mecab/dic/mecab-ipadic-neologd"
    tokenizer = mecab_tokenizer.Tokenizer("-Ochasen -d " + neologd_dic_path)
    token_list = tokenizer.tokenize(sentence)
    for token in token_list:
        tokenlist.append(token.surface)
        hinshilist.append(token.pos.split(",")[0])
    return hinshilist,tokenlist
def makeworddic(inputsentences,outputsentences):
    word_to_id = {"PAD":0,"UNK":1,"SOS":2,"EOS":3}
    id_to_word = {0:"PAD",1:"UNK",2:"SOS",3:"EOS"}
    #catogory:22ko+nanimonashi
    #hinshi:15(meishifukumu)
    unknumber = 0
    word_to_freq = {}
    word_to_part = {}
    for inputsentence in inputsentences:
        inputtokenlist = splitmecab(inputsentence)
        for token in inputtokenlist:
            if token not in word_to_freq:
                word_to_freq[token] = 1
            elif token in word_to_freq:
                word_to_freq[token] = word_to_freq[token]+1
    for outputsentence in outputsentences:
        outputtokenlist = splitmecab(outputsentence)
        for token in outputtokenlist:
            if token not in word_to_freq:
                word_to_freq[token] = 1
            elif token in word_to_freq:
                word_to_freq[token] = word_to_freq[token]+1
    print(len(word_to_freq))
    for word,count in word_to_freq.items():
        if count>unknumber:
            word_to_id[word] = len(word_to_id)
    for word in word_to_id:
        if word not in id_to_word.values():
            id_to_word[len(id_to_word)] = word
    #print(word_to_id)
    return word_to_id,id_to_word,word_to_freq
def makeworddicbynew(inputsentences,outputsentences):
    word_to_id = {"PAD":0,"SOS":1,"EOS":2,"RENTAIPAD":3,"SETTOUPAD":4,"MEISHIPAD":5,"DOUSHIPAD":6,"KEIYOUSHIPAD":7,"FUKUSHIPAD":8,"SETSUZOKUPAD":9,"JYOSHIPAD":10,"KANDOUSHIPAD":11,"KIGOUPAD":12,"FIRAPAD":13,"SONOTAPAD":14,"MITIGOPAD":15}
    id_to_word = {0:"PAD",1:"SOS",2:"EOS",3:"RENTAIPAD",4:"SETTOUPAD",5:"MEISHIPAD",6:"DOUSHIPAD",7:"KEIYOUSHIPAD",8:"FUKUSHIPAD",9:"SETSUZOKUPAD",10:"JYOSHIPAD",11:"KANDOUSHIPAD",12:"KIGOUPAD",13:"FIRAPAD",14:"SONOTAPAD",15:"MITIGOPAD"}
    #catogory:22ko+nanimonashi
    #hinshi:15(meishifukumu)
    unknumber = 0
    word_to_freq = {}
    word_to_part = {}
    word_to_hinshi = {}
    for inputsentence in inputsentences:
        inputhinshilist,inputtokenlist = splitmecab(inputsentence)
        for (token,hinshi) in zip(inputtokenlist,inputhinshilist):
            if token not in word_to_freq:
                word_to_hinshi[token] = hinshi
                word_to_freq[token] = 1
            elif token in word_to_freq:
                word_to_freq[token] = word_to_freq[token]+1
    for outputsentence in outputsentences:
        outputhinshilist,outputtokenlist = splitmecab(outputsentence)
        for (token,hinshi) in zip(outputtokenlist,outputhinshilist):
            if token not in word_to_freq:
                word_to_hinshi[token] = hinshi
                word_to_freq[token] = 1
            elif token in word_to_freq:
                word_to_freq[token] = word_to_freq[token]+1
    for word,count in word_to_freq.items():
        if count>unknumber:
            word_to_id[word] = len(word_to_id)
    for word in word_to_id:
        if word not in id_to_word.values():
            id_to_word[len(id_to_word)] = word
    #print(word_to_id)
    return word_to_id,id_to_word,word_to_freq,word_to_hinshi
def sentence_to_id(convsentencelist,word_to_id):
    training_data = []
    sentence_id_list = []
    input_id_list = []
    output_id_list = []
    inputlist = convsentencelist[0]
    outputlist = convsentencelist[1]
    for (input,output) in zip(inputlist,outputlist):
        setlist = []
        tokeninputlist = splitmecab(input)
        tokenoutputlist = splitmecab(output)
        inputidlist = []
        outputidlist = []
        for token in tokeninputlist:
            if token in word_to_id:
                inputidlist.append(word_to_id[token])
            elif token not in word_to_id:
                inputidlist.append(word_to_id["UNK"])
        for token in tokenoutputlist:
            if token in word_to_id:
                outputidlist.append(word_to_id[token])
            elif token not in word_to_id:
                outputidlist.append(word_to_id["UNK"])
        setlist.append(inputidlist)
        setlist.append(outputidlist)
        training_data.append(setlist)
    return training_data
def padding_sentence(sentencelist):
    maxlength = 0
    paddingnumberlist = []
    originallegth = []
    for sentence in sentencelist:
        originallegth.append(len(sentence))
        if len(sentence) > maxlength:
            maxlength = len(sentence)
    for sentence in sentencelist:
        while len(sentence)<maxlength:
            sentence.append(0)
    for sentence in sentencelist:
        paddingnumber = -1
        for i,index in enumerate(sentence):
            if index ==0:
                paddingnumber = i
                break
        paddingnumberlist.append(paddingnumber)
    return sentencelist,paddingnumberlist,originallegth
def predict_index_to_sentence(sentence,word_to_id):
    returnlist = []
    for token in splitmecab(sentence):
        if token in word_to_id:
            returnlist.append(word_to_id[token])
        else:
            returnlist.append(0)
    return returnlist
def outoutpredict(path,word_to_id,id_to_word,embed_size,hidden_size,BATCH_SIZE):
    the_model = EncoderDecoderbytoken(len(word_to_id),embed_size,hidden_size,BATCH_SIZE)
    the_model.load_state_dict(torch.load(path))
    with torch.no_grad():
         sentence = "初めまして。"
         idlist = predict_index_to_sentence(sentence,word_to_id)
         #print(idlist)
         idnolist=the_model.predict(idlist)
         returnlist = []
         for id in idnolist:
             returnlist.append(id_to_word[id])
         print(returnlist)
def main():
    miniinputpath = "./testinput.txt"
    minioutputpath = "./testoutput.txt"
    inputsentence,outputsentence = minidataload(miniinputpath,minioutputpath)
    word_to_id,id_to_word,word_to_freq= makeworddicbynew(inputsentence,outputsentence)
    #jumancheck("会社")
    #word_to_id,id_to_word,word_to_freq= makeworddic(inputsentence,outputsentence)
    #jumancheck("いままで五六ぺん見たことのある大きな三毛猫でした。")
    #print(splitmecab("いままで五六ぺん見たことのある大きな三毛猫でした。"))
    #print(word_to_id)
    """convsentencelist = []
    convsentencelist.append(inputsentence)
    convsentencelist.append(outputsentence)
    training_data =sentence_to_id(convsentencelist,word_to_id)
    embed_size = 10
    hidden_size = 15
    EPOCH_NUM = 1
    BATCH_SIZE = 2
    #predict("./testmodel",word_to_id,id_to_word,embed_size,hidden_size,BATCH_SIZE)
    encoderdecoder = EncoderDecoderbytoken(len(word_to_id),embed_size,hidden_size,BATCH_SIZE).to("cuda")
    optimizer = optim.Adam(encoderdecoder.parameters(), lr=0.01, weight_decay=1e-4)
    print("----------train-----------")
    for number,epoch in enumerate(range(EPOCH_NUM)):
        epoch_loss = 0
        print("------{}/{}--------".format(number,EPOCH_NUM))
        n = len(training_data)
        random.shuffle(training_data)
        batch_training_data = []
        for i in range(0,n,BATCH_SIZE):
            if i+BATCH_SIZE>n:
                batch_training_data.append(training_data[i:])
            else:
                batch_training_data.append(training_data[i:i+BATCH_SIZE])
        for convbatch in batch_training_data:
            encoderdecoder.zero_grad()
            inputlist = []
            outputlist = []
            convbatch.sort(key=lambda convbatch:len(convbatch[0]),reverse=True)
            for conv in convbatch:
                inputlist.append(conv[0])
                outputlist.append(conv[1])
            inputsentencelist,inputpaddingnumberlist,inputoriginallegth =padding_sentence(inputlist)
            outputsentencelist,outputpaddingnumberlist,outputoriginallegth =padding_sentence(outputlist)
            loss = encoderdecoder(inputsentencelist,inputpaddingnumberlist,inputoriginallegth,outputsentencelist,outputpaddingnumberlist)
            epoch_loss +=loss
            loss.backward()
            optimizer.step()
            break
        print(epoch_loss/embed_size)
    print("predict")
    with torch.no_grad():
        while True:
            input_ = input("INPUT>>>")
            sentence = input_
            idlist = predict_index_to_sentence(sentence,word_to_id)
            idnolist=encoderdecoder.predict(idlist)
            returnlist = []
            for id in idnolist:
                returnlist.append(id_to_word[id])
            print(returnlist)
    torch.save(encoderdecoder.state_dict(), "./testmodel")"""
if  __name__ == '__main__':
    print(torch.cuda.is_available())
    main()