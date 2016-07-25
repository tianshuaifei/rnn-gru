# -*- coding:UTF-8 -*-
import sys
import cPickle
import numpy as np
from utils import load_model_parameters_theano

fn = 'word2idx.pkl'
with open(fn, 'r') as f:                   
    idx2word=cPickle.load(f)
    f.close()       
fn1 = 'label2idx.pkl'
with open(fn1, 'r') as f1:                   
    idx2label=cPickle.load(f1)
    f1.close()
model = load_model_parameters_theano('./data/model.npz')    
def test():
    le=sys.argv
    del le[0]
    b=[]
    for x in le[:]:
        if x in idx2word.values():
            b.append( idx2word.values().index(x)) 
        else:
            b.append(len(idx2word)-1) 
    print b
    y=model.predict(b)
    label=[]
    for x in y[:]:
        samples1 = np.random.multinomial(1, x)
        sampled_word1 = np.argmax(samples1)
        label.append(idx2label[sampled_word1])
    print label    
if __name__ == '__main__':
    test()