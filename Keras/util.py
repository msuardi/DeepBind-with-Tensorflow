import tensorflow as tf
import numpy as np
import math
import random
from keras import backend as K

dictpad={'A':[1.,0.,0.,0.],'C':[0.,1.,0.,0.],'G':[0.,0.,1.,0.],'T':[0.,0.,0.,1.],'U':[0.,0.,0.,1.],'N':[0.25,0.25,0.25,0.25]}
dictReverse={'A':'T','C':'G','G':'C','T':'A','N':'N'} #dictionary to implement reverse-complement mode

#funzione per trasformare le sequenze di input in array, presente nel paper supplementare pag.3
def seqtopad(sequence,motlen):
    pad=[]
    for j in range(motlen-1):
        pad.extend(dictpad['N'])
    res=pad.copy()
    for i in range(len(sequence)):
        res.extend(dictpad[sequence[i]])
    res.extend(pad)
    return np.asarray(res)

#funzione per trovare filamento opposto della sequenza
def reverse(sequence):
    revseq=''
    for i in sequence:
        revseq+=dictReverse[i]
    return revseq    

#funzioni per inizializzare momentum_rate e learning_rate, paper suppl. pag. 12
def logsampler(a,b,tensor=0):
    if(tensor==1):
        x=tf.Variable(tf.random_uniform([1],minval=0,maxval=1))
    else:
        x=np.random.uniform(low=0,high=1)
    y=10**((math.log10(b)-math.log10(a))*x + math.log10(a))
    return y

def sqrtsampler(a,b,tensor=0):
    if(tensor==1):
        x=tf.Variable(tf.random_uniform([1],minval=0,maxval=1))
    else:
        x=np.random.uniform(low=0,high=1)
    y=(b-a)*math.sqrt(x)+a
    return y

#funzione per effettuare il padding delle sequenze nel caso in cui gli input hanno sequenze di lunghezza diversa
#necessario per RNAcompete, anche perché i tensori accettano input della stessa forma
def padsequence(sequence,maxlength):
    return sequence + 'N'*(maxlength-len(sequence))
    
#funzione di perdita per CHIP-seq e SELEX, non è di default in Kears
#TODO va in loop 
def log_loss(label,prediction):
    def sigma(x):
        return 1/(1+math.e**(-x))
    return K.mean(-label*K.log(sigma(prediction)) - (1-label)*K.log(1-sigma(prediction)))

#funzione per creare le sequenze di training con specificità 0 nel caso di CHIP e SELEX (forniti solo quelli "positivi")
def dinucshuffle(sequence):
    b=[sequence[i:i+2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d=''.join([str(x) for x in b])
    return d
    
