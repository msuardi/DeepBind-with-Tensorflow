import random
import gzip
import csv
import numpy as np
from util import seqtopad,dinucshuffle

#funzione per aprire il file di training di CHIP-seq
def openChip(sequencefile):
    exc=False
    listExc=['RAD21','ZNF274','ZNH143','SIX5'] #nel paper sono delle eccezioni per la motiflen, provato sperimentalmente
    for elem in listExc:
        if elem in sequencefile:
            exc=True
            break
    motiflen=24 if not(exc) else 32
    train_dataset=[]
    with gzip.open(sequencefile, 'rt') as data:
        next(data)
        reader = csv.reader(data,delimiter='\t')
        for row in reader:
                train_dataset.append([seqtopad(row[2],motiflen),1])
                train_dataset.append([seqtopad(dinucshuffle(row[2]),motiflen),0])
    random.shuffle(train_dataset)
    train_seq=np.asarray([elem[0] for elem in train_dataset])
    train_lab=np.asarray([elem[1] for elem in train_dataset])
    return train_seq,train_lab,motiflen

#funzione per aprire il file di test di CHIP-seq
def openChipTest(sequencefile):
    exc=False
    listExc=['RAD21','ZNF274','ZNH143','SIX5']
    for elem in listExc:
        if elem in sequencefile:
            exc=True
            break
    motiflen=24 if not(exc) else 32
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
