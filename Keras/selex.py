import gzip
import random
import csv
import numpy as np
from util import *

#funzione per ottenere dal nome del file la lunghezza della sequenza e ritornare la corrispondente motiflen
def getMotiflenSelex(filename):
    fields=filename.split('_')
    res=''
    for i in fields[2]:
        if i.isdigit():
            res=res+i 
    if int(res)==30:
        return 24
    elif int(res)==40:
        return 32
    else:
        return int(res)

#funzione per aprire il file di training di Selex
def openSelex(trainfile,motiflen):
    train_dataset=[]
    with gzip.open(trainfile, 'rt') as data:
        next(data)
        reader = csv.reader(data,delimiter='\t')
        for row in reader:
                train_dataset.append([seqtopad(row[2],motiflen),1])
                train_dataset.append([seqtopad(dinucshuffle(row[2]),motiflen),0])
    random.shuffle(train_dataset) #migliora le performance?
    train_seq=np.asarray([elem[0] for elem in train_dataset])
    train_lab=np.asarray([elem[1] for elem in train_dataset])
    return train_seq,train_lab
    
#funzione per aprire il file di test di Selex
def openSelexTest(sequencefile,motiflen):
    test_dataset=[]
    with gzip.open(sequencefile, 'rt') as data:
        next(data)
        reader = csv.reader(data,delimiter='\t')
        for row in reader:
            test_dataset.append([seqtopad(row[2],motiflen),int(row[3])])
    random.shuffle(test_dataset)
    test_seq=np.asarray([elem[0] for elem in test_dataset])
    test_lab=np.asarray([elem[1] for elem in test_dataset])
    return test_seq,test_lab    