import csv
import math
import numpy as np
import random
import itertools
from util import *
from scipy.stats import pearsonr,spearmanr
from sklearn import metrics

basesRNA='ACGU'

#funzione per creare test e training
def openRNA(sequencefile,targetfile,tfChoose):
    motiflen=16
    train_lab=[]
    clear_lab=[]
    train_dat=[]
    test_dat=[]
    train_data=[]
    test_data=[]
    maxlenA=0
    maxlenB=0

    with open(targetfile,'r') as data:
        intest=data.readline().split()
        col = intest.index(tfChoose) #si sceglie un determinato fattore di trascrizione
        reader=csv.reader(data, delimiter='\t')
        for row in reader:
            if math.isnan(np.float(row[col]))==False: #alcune entry sono nan
                train_lab.append(np.float(row[col]))
            else:
                clear_lab.append(reader.line_num) #salvo l'indice di queste entry per non considerare quelle sequenze
    with open(sequencefile,'r') as data2:
        next(data2)
        count=0
        reader=csv.reader(data2,delimiter='\t')
        for row in reader: #divido A e B come training e test set
            if row[0]=='A' and reader.line_num not in clear_lab:
                train_dat.append((row[2],train_lab[count]))
                count+=1
                if len(row[2])>maxlenA:
                    maxlenA=len(row[2])
            elif row[0]=='B' and reader.line_num not in clear_lab:
                test_dat.append((row[2],train_lab[count]))
                count+=1                
                if len(row[2])>maxlenB:
                    maxlenB=len(row[2]) #salvo la massima lunghezza per fare il padding
            
    #ciclo per fare padding in training set   
    for i in range(len(train_dat)):
        if len(train_dat[i][0])!=maxlenA:
            train_data.append((seqtopad(padsequence(train_dat[i][0],maxlenA),motiflen),train_dat[i][1]))
        else:
            train_data.append((seqtopad(train_dat[i][0],motiflen),train_dat[i][1]))
    #ciclo per fare padding in test set
    for j in range(len(test_dat)):
        if len(test_dat[j][0])!=maxlenB:
            test_data.append((seqtopad(padsequence(test_dat[j][0],maxlenB),motiflen),test_dat[j][1]))
        else:
            test_data.append((seqtopad(test_dat[j][0],motiflen),test_dat[j][1]))
            
    return train_data,test_data,test_dat,motiflen
            
#funzione per creare il validation set come frazione del training set
def getValidRNA(traindataset,perc):
    random.shuffle(traindataset)
    frac=int(len(traindataset)*perc)
    return traindataset[:frac],traindataset[frac:]

#funzione per ottenere lo zscore: per ognuna delle sequenze di 7-meri salvo i valori predetti e originali delle specificità delle sequenze contenenti
#come sottosequenza quel 7-mero; faccio poi la mediana di ciascun array
def Zscore(sequences,valuesOrig,valuesPred,listSeven):
    orig=[]
    pred=[]
    for l in listSeven:
        tempOrig=[]
        tempPred=[]
        for i in range(len(sequences)):
            if l in sequences[i]:
                tempOrig.append(valuesOrig[i])
                tempPred.append(valuesPred[i])
        orig.append(tempOrig)
        pred.append(tempPred)
        
    zscOrig=[np.median(elem) for elem in orig if len(elem)!=0]
    zscPred=[np.median(elem) for elem in pred if len(elem)!=0]
    return zscOrig,zscPred

#funzione per ritornare array per auc
def aucscore(sequences,lista):
    aucl=np.array([l in seq for l in lista for seq in sequences])
    auclc=np.array([aucl[i:i+len(sequences)] for i in range(0,len(sequences),len(sequences))])
    return auclc

#funzione per generare la lista di tutti i 7-meri di RNA, serve per misure di accuratezza di RNACompete
def gen7list():
    lista=[''.join(elem)for elem in itertools.product(basesRNA,repeat=7)]
    return lista

#TODO check if all is ok with parallel implementation
#rappresenta le statistiche da fare con RNAcompete, seguendo il paper
def statsRNA(orig_dataset,predictionarray,testlab):
    listSev=gen7list()
    sequences=[elem[0] for elem in orig_dataset]
    values=[elem[1] for elem in orig_dataset]
    zscoreorig,zscorepred=Zscore(sequences,values,predictionarray)
    aucSc=aucscore(sequences,listSev)
    coeffNorm=(pearsonr(predictionarray,testlab),spearmanr(predictionarray,testlab))
    coeffZ=(pearsonr(zscoreorig,zscorepred),spearmanr(zscoreorig,zscorepred))
    indYes=np.asarray([elem*predictionarray for elem in aucSc])
    indYesSort=np.asarray([np.argsort(elem)[(len(elem)/2):] for elem in indYes])
    indNo=np.asarray([np.invert(elem)*predictionarray for elem in aucSc])
    indNoSort=np.asarray([np.argsort(elem)[(len(elem)/2):] for elem in indNo])
    for i in len(indYesSort):
        indYesSort[i].extend(indNoSort[i])
    pred=[]
    lab=[]
    for j in len(indYesSort):
        temp=[values[i] for i in indYesSort[j]]
        templ=[aucSc[i] for i in indYesSort[j]]
        pred.append(temp)
        lab.append(templ)
    pred=np.asarray(pred)
    lab=np.asarray(lab)
    auc=[]
    for i in len(pred):
        auc.append(metrics.auc(pred[i],lab[i]))
    eScore=auc-np.repeat(0.5,len(auc))
    return auc,eScore,coeffNorm,coeffZ
    
#TODO testing in parallel
#funzione per effettuare il testing su dati in vivo, presenti in una sottocartella di RNACompete
def testRnaVivo(sequencefile,motiflen,maxlenseq):
    with open(sequencefile,'r') as data:
        lines=data.readlines()
        test=[]
        for line in lines:
            temp=[]
            line=line.strip('\n').upper().split('\t')
            linetolist=[line[1][i:i+maxlenseq] for i in range(len(line[1])-maxlenseq)]
            for elem in linetolist:
                temp.append(seqtopad(elem,motiflen))
        test.append(temp)
    return np.asarray(test)

