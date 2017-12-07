import csv
import random
import numpy as np
from util import seqtopad,reverse

def openValidPBM(filename,name,motlen): #load validation set
    resseq=[]
    with open(filename,'r') as data:
        next(data)
        reader=csv.reader(data,delimiter='\t')
        for line in reader:
            if line[0]==name:
                if len(resseq)==0:
                    arrayType=line[1]
                resseq.append((seqtopad(line[2],motlen),np.float(line[3])))
    random.shuffle(resseq) 
    resdat=[resseq[i][0] for i in range(len(resseq))]
    reslab=[resseq[i][1] for i in range(len(resseq))]
    return np.asarray(resdat),np.asarray(reslab),arrayType
    
def getValidPBM(filename,tfName,motlen): #create validation set
    validseq,validlab,arraytype=openValidPBM(filename,tfName,motlen)
    validstd=np.std(validlab)
    validavg=np.average(validlab)
    validlab=(validlab - validavg)/validstd 
    validseq=np.reshape(validseq,[validseq.shape[0],validseq.shape[1],1])
    return validseq,validlab,validstd,validavg,arraytype

def openPBM(filename,motlen,reverseMode=False): #creates a dict where for every TF there's an array with sequences-specificities
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
                    last=row[0]
            resseq.append((seqtopad(row[2],motlen),np.float(row[3])))
            if reverseMode:
                resseq.append((seqtopad(reverse(row[2]),motlen),np.float(row[3])))
    return diz

def predictSequencesPBM(filename,arrayType,motlen): #creates list of sequences to predict (HK or ME)
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
