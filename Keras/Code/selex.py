import gzip
import random
import csv
import numpy as np
from util import seqtopad,dinucshuffle

def getMotiflenSelex(filename): #get name of TF and return motiflen (supplementary paper p.20)
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

def openSelex(trainfile,motiflen): #open Selex training set
    train_dataset=[]
    with gzip.open(trainfile, 'rt') as data:
        next(data)
        reader = csv.reader(data,delimiter='\t')
        for row in reader:
                train_dataset.append([seqtopad(row[2],motiflen),1])
                train_dataset.append([seqtopad(dinucshuffle(row[2]),motiflen),0])
    random.shuffle(train_dataset)
    train_seq=np.asarray([elem[0] for elem in train_dataset])
    train_lab=np.asarray([elem[1] for elem in train_dataset])
    return train_seq,train_lab
    
def openSelexTest(sequencefile,motiflen): #open Selex Test
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
