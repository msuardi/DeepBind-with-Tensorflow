import csv
import random
import numpy as np
import time
from util import *

#funzione per caricare il validation set, ovvero il file con 66 TF di un solo tipo 
def openValid(filename,name,motlen):
    resseq=[]
    with open(filename,'r') as data:
        next(data)
        reader=csv.reader(data,delimiter='\t')
        for line in reader:
            if line[0]==name:
                if len(resseq)==0:
                    arrayType=line[1]
                resseq.append((seqtopad(line[2],motlen),np.float(line[3])))
    random.shuffle(resseq) #migliora le performance?
    resdat=[resseq[i][0] for i in range(len(resseq))]
    reslab=[resseq[i][1] for i in range(len(resseq))]
    return np.asarray(resdat),np.asarray(reslab),arrayType
    
#funzione che crea il validation set
def createValidation(filename,tfName,motlen):
    validseq,validlab,arraytype=openValid(filename,tfName,motlen)
    validstd=np.std(validlab)
    validavg=np.average(validlab)
    validlab=(validlab - validavg)/validstd #migliora le performance?
    validseq=np.reshape(validseq,[validseq.shape[0],validseq.shape[1],1])
    return validseq,validlab,validstd,validavg,arraytype


#funzione per creare un dizionario dove per ogni fattore di trascrizione c'è un array contenente coppie sequenza-specificità
def openPBM(filename,motlen,reverseMode=False):
    diz={}
    resseq=[]
    last=''
    with open(filename,'r') as data:
        next(data)
        reader=csv.reader(data,delimiter='\t')
        for row in reader:
            if row[0]!=last:
                    if last!='':
                        random.shuffle(resseq)
                        diz[last]=resseq
                        resseq=[]
                        print('passi fatti ',len(diz.keys())) #per avere il controllo, operazioe pesante
                    last=row[0]
            resseq.append((seqtopad(row[2],motlen),np.float(row[3])))
            if reverseMode:
                resseq.append((seqtopad(reverse(row[2]),motlen),np.float(row[3])))
    return diz

#funzione che crea la lista delle sequenze per cui deve essere predetta la specificità (sono sempre le stesse)
def predictSequences(filename,arrayType,motlen):
    with open(filename,'r') as data:
        resseq=[]
        next(data)
        reader=csv.reader(data,delimiter='\t')
        for line in reader:
            if line[1]==arrayType:
                resseq.append(seqtopad(line[2],motlen))
            elif len(resseq)!=0:
                break
    resseq=np.asarray(resseq)
    testseq=np.reshape(resseq,[resseq.shape[0],resseq.shape[1],1])    
    return testseq

