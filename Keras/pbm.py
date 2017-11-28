import csv
import random
import numpy as np
from util import seqtopad,reverse
import gzip
import math


#funzione per caricare il validation set, ovvero il file con 66 TF di un solo tipo 
def openValidPBM(filename,name,motlen):
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
def getValidPBM(filename,tfName,motlen):
    validseq,validlab,arraytype=openValidPBM(filename,tfName,motlen)
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
                        print('passi fatti ',len(diz.keys())) #per avere il controllo, operazione pesante
                    last=row[0]
            resseq.append((seqtopad(row[2],motlen),np.float(row[3])))
            if reverseMode:
                resseq.append((seqtopad(reverse(row[2]),motlen),np.float(row[3])))
    return diz

#funzione che crea la lista delle sequenze per cui deve essere predetta la specificità (sono sempre le stesse)
def predictSequencesPBM(filename,arrayType,motlen):
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

#TODO tutte le sottosequenze, fare la media
def openTestPBM(filename,motlen):
    resseq=[]
    with open(filename,'r') as data:
        reader=csv.reader(data,delimiter='\t')
        for line in reader:
            resseq.append((seqtopad(line[0][:40],motlen),np.int32(line[1])))
    return resseq
    
def openPBMrev(sequencefile,targetfile,tfChoose):
    motiflen=24
    train_lab=[]
    clear_lab=[]
    train_dat=[]
    test_dat=[]

    with gzip.open(targetfile, 'rt') as data:
        intest=data.readline().split()
        col = intest.index(tfChoose) #si sceglie un determinato fattore di trascrizione
        reader=csv.reader(data, delimiter='\t')
        for row in reader:
            if math.isnan(np.float(row[col]))==False: #alcune entry sono nan
                train_lab.append(np.float(row[col])/np.average(np.array(row).astype(float)))
            else:
                clear_lab.append(reader.line_num) #salvo l'indice di queste entry per non considerare quelle sequenze
    with gzip.open(sequencefile, 'rt') as data:
        next(data)
        count=0
        reader=csv.reader(data,delimiter='\t')
        for row in reader: #divido A e B come training e test set
            if row[0]=='A' and reader.line_num not in clear_lab:
                train_dat.append((seqtopad(row[2],motiflen),train_lab[count]))
                count+=1
            elif row[0]=='B' and reader.line_num not in clear_lab:
                test_dat.append((seqtopad(row[2],motiflen),train_lab[count]))
                count+=1                
                        
    return train_dat,test_dat,motiflen
    
def scoretobin(score):
    threshold=np.average(score)+4*np.std(score)
    res= np.array([1 if elem>threshold else 0 for elem in score])
    return res
    
            
