import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import MeCab
import mecab_tokenizer
import re
import xml.etree.ElementTree as ET
import gensim
model_dir = './entity_vector/entity_vector.model.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_dir, binary=True)
word_vectors = model.wv
weights = word_vectors.syn0
similar_words = model.wv.most_similar(positive=["きょとん"], topn=9)
print(*[" ".join([v, str("{:.2f}".format(s))]) for v, s in similar_words], sep="\n")