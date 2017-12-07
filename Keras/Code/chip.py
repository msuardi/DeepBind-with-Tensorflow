import random
import gzip
import csv
import numpy as np
from util import seqtopad,dinucshuffle

def exceptchip(name):
    listExc=['RAD21','ZNF274','ZNH143','SIX5'] #following supplementary paper p.18
    for elem in listExc:
        if elem in name:
            return 32
    return 24
    
def openChip(sequencefile): #open chip training file
    motiflen=exceptchip(sequencefile)
    train_dataset=[]
    with gzip.open(sequencefile, 'rt') as data:
        next(data)
        reader = csv.reader(data,delimiter='\t')
        for row in reader:
                train_dataset.append([seqtopad(row[2],motiflen),1])
                train_dataset.append([seqtopad(dinucshuffle(row[2]),motiflen),0]) #add also background sequence
    random.shuffle(train_dataset) #reduce overfitting
    train_seq=np.asarray([elem[0] for elem in train_dataset])
    train_lab=np.asarray([elem[1] for elem in train_dataset])
    return train_seq,train_lab,motiflen

def openChipTest(sequencefile): #open chip test file
    motiflen=exceptchip(sequencefile)
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
