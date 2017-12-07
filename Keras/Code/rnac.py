import csv
import math
import numpy as np
import random
import itertools
from util import seqtopad,padsequence
from scipy.stats import pearsonr,spearmanr
from sklearn import metrics

basesRNA='ACGU'

def openRNA(sequencefile,targetfile,tfChoose): #create training and test set for RNA
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
        col = intest.index(tfChoose) #choose a specific TF
        reader=csv.reader(data, delimiter='\t')
        for row in reader:
            if math.isnan(np.float(row[col]))==False: #some entry NaN
                train_lab.append(np.float(row[col]))
            else:
                clear_lab.append(reader.line_num) #save not-NaN entries
    with open(sequencefile,'r') as data2:
        next(data2)
        count=0
        reader=csv.reader(data2,delimiter='\t')
        for row in reader: #A is training, B is test
            if row[0]=='A' and reader.line_num not in clear_lab:
                train_dat.append((row[2],train_lab[count]))
                count+=1
                if len(row[2])>maxlenA:
                    maxlenA=len(row[2])
            elif row[0]=='B' and reader.line_num not in clear_lab:
                test_dat.append((row[2],train_lab[count]))
                count+=1                
                if len(row[2])>maxlenB:
                    maxlenB=len(row[2]) #save max length for padding
            
    for i in range(len(train_dat)): #padding in training set
        if len(train_dat[i][0])!=maxlenA:
            train_data.append((seqtopad(padsequence(train_dat[i][0],maxlenA),motiflen),train_dat[i][1]))
        else:
            train_data.append((seqtopad(train_dat[i][0],motiflen),train_dat[i][1]))

    for j in range(len(test_dat)): #padding in test set
        if len(test_dat[j][0])!=maxlenB:
            test_data.append((seqtopad(padsequence(test_dat[j][0],maxlenB),motiflen),test_dat[j][1]))
        else:
            test_data.append((seqtopad(test_dat[j][0],motiflen),test_dat[j][1]))
            
    return train_data,test_data,test_dat,motiflen
            
def getValidRNA(traindataset,perc): #create validation set
    random.shuffle(traindataset)
    frac=int(len(traindataset)*perc)
    return traindataset[:frac],traindataset[frac:]

def Zscore(sequences,valuesOrig,valuesPred,listSeven): #obtain z-score, supplementary paper p.15
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
        
    zscOrig=np.array([np.median(elem) for elem in orig if len(elem)!=0])
    zscOrig=(zscOrig-np.average(zscOrig))/np.std(zscOrig)
    zscPred=[np.median(elem) for elem in pred if len(elem)!=0]
    zscPred=(zscPred-np.average(zscPred))/np.std(zscPred)
    return zscOrig,zscPred

def aucscore(sequences,values,lista): #aucscore list
    aucl=np.array([l in seq for l in lista for seq in sequences]) 
    auclist=[calc_auc(values,aucli) for aucli in aucl]
    return auclist,aucl

def gen7list(): #generate all 7-mers
    lista=[''.join(elem)for elem in itertools.product(basesRNA,repeat=7)]
    return lista

def statsRNA(orig_dataset,predictionarray,testlab): #generate stats, supplementary paper p.15
    listSev=gen7list()
    sequences=[elem[0] for elem in orig_dataset]
    values=[elem[1] for elem in orig_dataset]
    zscoreorig,zscorepred=Zscore(sequences,values,predictionarray,listSev)
    aucSc,yesornot=aucscore(sequences,values,listSev)
    coeffNorm=(pearsonr(predictionarray,testlab),spearmanr(predictionarray,testlab))
    coeffZ=(pearsonr(zscoreorig,zscorepred),spearmanr(zscoreorig,zscorepred))
    valYes=np.asarray([[values[i] for i in range(len(values))] for sev in listSev if sev in sequences[i]])
    indYesSort=np.asarray([np.argsort(el) for el in valYes])
    yesornotIN=np.asarray([yesornot[i][indYesSort[i]] for i in range(len(indYesSort))])
    yesornotINspl=np.asarray([elem[(len(elem)/2):] for elem in yesornotIN])
    valYes=np.asarray([valYes[i][indYesSort[i]] for i in range(len(indYesSort))])
    valYesspl=np.asarray([elem[(len(elem)/2):] for elem in valYes]) 
    valNo=np.asarray([values[i] for i in range(len(values)) for sev in listSev if sev not in sequences[i]])    
    indNoSort=np.asarray([np.argsort(el) for el in valNo])
    yesornotNOT=np.asarray([yesornot[i][indNoSort[i]] for i in range(len(indNoSort))])
    yesornotNOTspl=np.asarray([elem[(len(elem)/2):] for elem in yesornotNOT])
    valNo=np.asarray([valNo[i][indNoSort[i]] for i in range(len(indNoSort))])
    valNospl=np.asarray([elem[(len(elem)/2):] for elem in valNo])
    intensities=np.asarray([valYesspl[i].extend(valNospl[i]) for i in range(len(valYesspl))])
    labels=np.asarray(yesornotINspl[i].extend(yesornotNOTspl[i]) for i in range(len(yesornotINspl))) 
    aucscore=np.asarray([calc_auc(intensities[i],labels[i]) for i in range(len(intensities))])
    escore=aucscore-0.5

    return coeffNorm,coeffZ,aucscore,escore
    
def testRNAVivo(sequencefile,motiflen,maxlenseq): #test in vivo sequences
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
