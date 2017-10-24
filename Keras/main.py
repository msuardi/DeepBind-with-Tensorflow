from __future__ import print_function
from keras.models import Model,load_model,save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling1D, Conv1D, AveragePooling1D,Input
from keras import optimizers
from keras.callbacks import EarlyStopping,TensorBoard
from keras.utils import plot_model
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from sklearn.metrics import roc_curve,auc
from os import listdir,remove
from os.path import join
import pickle
import csv

import random
import numpy as np
from util import *
from pbm import *
from rnac import *
from chip import *
from selex import *
from multi_gpu import *
import time
import math

#from multi_gpu import *
#import tensorflow as tf

#inserisci queste due linee all'inizio e alla fine di ogni funzione di cui si vuole verificare il tempo di esecuzione
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))

batch_size = 64 #fissata sempre, su deepfind dovrebbe essere più alta
epochs = 50 #numero di cicli di training, settabile a piacere -> più è alta più dovrebbe essere accurato il training
bases='ACGT'
basesRNA='ACGU'
otherArray={'ME':'HK', 'HK':'ME'} #dizionario per ottenere in PBM le sequenze da predire 
dropoutList=[0.5,0.75,1.0] #lista dei possibili valori di dropout
dropoutChoice=random.choice(dropoutList) #estrazione di uno dei valori a caso, dovrebbe essere estratto tramite Bernoulli

#determino learning_rate e momentum_rate estraendo a caso seguendo i due sampler definiti in util
learning_rate=logsampler(0.005,0.05)
momentum_rate=sqrtsampler(0.95,0.99)


#definisco la rete neurale per tutti gli esperimenti
def kerasNet(inpShape,motiflen,exp,hidd,sigmoid):
    inputs=Input(shape=(inpShape,1)) #layer di Input
    conv=Conv1D(filters=16,kernel_size=motiflen*4,activation='relu',strides=4)(inputs) #convoluzione 1 dimensionale con strides=4
    pool=MaxPooling1D(pool_size=int(conv.shape[1]))(conv) #maxpooling
    if exp=='RNA': #se l'esperimento è RNACompete si deve fare concatenazione alternata (ancora da fare) con average
        avgpool=AveragePooling1D(pool_size=int(conv.shape[1]))(conv)
        pool=AltConcatenate()([avgpool,pool])
        print(pool)
    flat=Flatten()(pool) 
    dropout=Dropout(dropoutChoice)(flat) #dropout con la probabilità
    if hidd==True: #hidden layer si o no? migliora le prestazioni?
        dropout=Dense(32, activation='relu',use_bias=True)(dropout) #32 hidden layer con relu
    if sigmoid==True:
        actfun='sigmoid'
    else:
        actfun='linear'
    neu=Dense(1, activation=actfun,use_bias=True)(dropout) #passo di rete neurale
    model=Model(inputs=inputs,outputs=neu) #creo il modello collegando l'input all'output
    print (model.summary())
    return model
    

#################################################################################################################
#PBM    
#TODO PBM with data downloaded and test chip in vivo with AUC

def predictPBM(fileTrain,fileValid,predTF,exp='DNA'):
    motiflen=24 #motiflen per esperimento PBM
    bestScore=0 #salvo il bestScore e il bestModel per il testing
    bestModel=None 
    bestTf=''
    validSeq,validLab,validStdDev,validAvg,arrayType=getValidPBM(fileValid,predTF,motiflen) #apre il validation set 
    print('Validation Done')
    dictionary=openPBM(fileTrain,motiflen,True) #apre il training set
    print('I have created the dictionary of %s' %(otherArray[arrayType]))
    listTf=list(dictionary.keys())
    predSeq=predictSequencesPBM(fileTrain,arrayType,motiflen) #crea le sequenze da predire
    print('I have collected the predict sequences, entering in the predict cycle...')
    for tfact in listTf: #per ogni fattore di trascrizione in input creo il modello, faccio fitting e valuto la prestazione sul validation
        trainSeq=np.asarray([elem[0] for elem in dictionary[tfact]]) #estraggo solo le sequenze
        trainLab=np.asarray([elem[1] for elem in dictionary[tfact]]) #estratto solo le etichette
        trainLabNorm=(trainLab - np.average(trainLab))/np.std(trainLab) #normalizzo le etichette
        trainSeq=np.reshape(trainSeq,[trainSeq.shape[0],trainSeq.shape[1],1]) #reshape necessario per il training
        print('Training data ready, with TF: %s' %(tfact))        
        
        model = kerasNet(int(trainSeq.shape[1]),motiflen,exp,True,False)
        #model = make_parallel(model,4)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=learning_rate,momentum=momentum_rate,nesterov=True,decay=1e-6),
                      metrics=['mae'])
    
        model.fit(trainSeq, trainLabNorm,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(validSeq, validLab))
        score = model.evaluate(validSeq, validLab, verbose=1)
        print('Model is %s, best, score is %f' % (tfact,score[0]))
        if score[0]<bestScore or bestScore==0:
            bestScore=score[0]
            bestModel=model
            bestTf=tfact
        print(bestTf)
    print('the Best model is ', bestTf)
    prediction = bestModel.predict(predSeq,batch_size,verbose=1)
    prediction = prediction*validStdDev + validAvg
    return prediction
    
def predictPBMrev(seqTrain,targetTrain,fileTest,predTF,exp='DNA'):
    train,valid,motiflen=openPBMrev(seqTrain,targetTrain,predTF)
    test=openTestPBM(fileTest,motiflen)
    testSeq=np.asarray([elem[0] for elem in test])
    testLab=np.asarray([elem[1] for elem in test])
    trainSeq=np.asarray([elem[0] for elem in train])
    trainLab=np.asarray([elem[1] for elem in train])
    validSeq=np.asarray([elem[0] for elem in valid])
    validLab=np.asarray([elem[1] for elem in valid])
    trainLab=(trainLab-np.average(trainLab))/np.std(trainLab)
    validLab=(validLab-np.average(validLab))/np.std(validLab) 
    trainSeq=np.reshape(trainSeq,[trainSeq.shape[0],trainSeq.shape[1],1])
    validSeq=np.reshape(validSeq,[validSeq.shape[0],validSeq.shape[1],1])
    testSeq=np.reshape(testSeq,[testSeq.shape[0],testSeq.shape[1],1])
    model = kerasNet(int(trainSeq.shape[1]),motiflen,exp,False,False)
    model = to_multi_gpu(model,4)    
    model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=learning_rate,momentum=momentum_rate,nesterov=True,decay=1e-6),
                      metrics=['mae'])
    
    model.fit(trainSeq, trainLab,
                  batch_size=batch_size*4,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(validSeq, validLab))
    score=model.predict(testSeq,batch_size,verbose=1)
    fpr, tpr, _ = roc_curve(testLab,score)
    roc_auc = auc(fpr, tpr)
    return score,roc_auc
    
    
#TODO per AUC considera positivi quelli che stanno 4 dev standard sopra la media
##############################################################################################################################
    
#rnacompete
#predizione delle sequenze di RNAcompete
def predictRNA(sequencefile,targetfile,tfChoose,perc,exp,invivo=''):
    train_dataset,test_dataset,orig_test_dataset,motiflen=openRNA(sequencefile,targetfile,tfChoose) #carica i dati RNAc training
    print('Train and test dataset OK')
    train_dat,valid_dat=getValidRNA(train_dataset,perc) #carica i dati RNAc validation
    print('Train and validation separation OK')
    train_seq=np.asarray([elem[0] for elem in train_dat]) 
    train_seq=np.reshape(train_seq,[train_seq.shape[0],train_seq.shape[1],1])
    train_lab=np.asarray([elem[1] for elem in train_dat])
    train_stddev=np.std(train_lab)
    train_avg=np.average(train_lab)    
    train_lab_norm=(train_lab-train_avg)/train_stddev #normalizzo le etichette
    valid_seq=np.asarray([elem[0] for elem in valid_dat]) 
    valid_seq=np.reshape(valid_seq,[valid_seq.shape[0],valid_seq.shape[1],1])
    valid_lab=np.asarray([elem[1] for elem in valid_dat])
    valid_lab_norm=(valid_lab-np.average(valid_lab)/np.std(valid_lab)) #normalizzo il validation set
    test_seq=np.asarray([elem[0] for elem in test_dataset]) 
    test_seq=np.reshape(test_seq,[test_seq.shape[0],test_seq.shape[1],1])
    test_lab=np.asarray([elem[1] for elem in test_dataset])
    model = kerasNet(int(train_seq.shape[1]),motiflen,exp,True,False)
    #train_lab_sta=normalize(train_lab)
    print('Construction ok Keras network OK')
    #model= make_parallel(model,4)
    model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=0.0001,momentum=0.95,nesterov=True,decay=1e-6),
                      metrics=['mae'])
    print('Ready to Fit OK')
    callbacks=[TensorBoard(log_dir='./', histogram_freq=0,  
          write_graph=True, write_images=True)]
    model.fit(train_seq, train_lab_norm,
                  batch_size=batch_size,#*4,
                  epochs=epochs,
                  verbose=1,callbacks=callbacks,
                  validation_data=(valid_seq, valid_lab_norm))
    print('Fit complete. Now score and prediction')
    if invivo!='': #faccio testing su dati in vivo
        prediction_res=[]
        test_seq2=testRNAVivo(invivo,16,41)
        for testseq in test_seq2:
            test_seq=np.reshape(test_seq,[test_seq.shape[0],test_seq.shape[1],1])
            prediction = model.predict(test_seq,batch_size,verbose=1)
            prediction_res.append(np.average(prediction))
        return prediction_res
        
    score = model.evaluate(test_seq, test_lab, verbose=1)
    prediction = model.predict(test_seq,batch_size,verbose=1)
    prediction = prediction*np.std(test_lab) + np.average(test_lab)
    prediction = np.reshape(prediction,[prediction.shape[0]]) 
    #statsRNA(orig_test_dataset,prediction,test_lab)
    return score,prediction,model
        
##########################################################################################################################
#ENCODECHIP

#predizioni di dati chipseq  
def predictChip(trainfile,testfile): 
    train_seq, train_lab,motiflen=openChip(trainfile) #estraggo dati di training
    #train_lab2=keras.utils.to_categorical(train_lab,num_classes=2)
    test_seq,test_lab=openChipTest(testfile) #estraggo dati di test
    #test_lab2=keras.utils.to_categorical(test_lab,num_classes=2)
    train_seq=np.reshape(train_seq,[train_seq.shape[0],train_seq.shape[1],1])
    test_seq=np.reshape(test_seq,[test_seq.shape[0],test_seq.shape[1],1])
    
    model = kerasNet(int(train_seq.shape[1]),motiflen,'DNA',False,True)
    model= to_multi_gpu(model,4)
    model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=learning_rate,momentum=momentum_rate,nesterov=True,decay=1e-6),metrics=['binary_accuracy'])
    print('Ready to Fit OK') 
    callbacks=[EarlyStopping(monitor='loss', min_delta=math.nan, patience=1)]#,TensorBoard(log_dir='./', histogram_freq=0,  
         # write_graph=True, write_images=True)]
    model.fit(train_seq, train_lab,
                  batch_size=batch_size*4,
                  epochs=epochs,#callbacks=callbacks,
                  verbose=1)
    print('Fit complete. Now score and prediction')
    
    score = model.evaluate(test_seq, test_lab, batch_size=batch_size,verbose=1)
    prediction = model.predict(test_seq,batch_size,verbose=1)
    prediction = np.reshape(prediction,[prediction.shape[0]]) 
    fpr, tpr, _ = roc_curve(test_lab,prediction)
    roc_auc = auc(fpr, tpr)
    return score,prediction,roc_auc,model
        
#works with msq, log_loss goes to nan always...
############################################################
#SELEX

def predictSelex(trainfile,testfile):
    motiflen=getMotiflenSelex(trainfile) #estraggo dal nome del file la lunghezza delle sequenze
    train_seq, train_lab=openSelex(trainfile,motiflen) #estraggo i dati di training
    #train_lab2=keras.utils.to_categorical(train_lab,num_classes=2)
    test_seq,test_lab=openSelexTest(testfile,motiflen) #estraggo i dati di testing
    #test_lab2=keras.utils.to_categorical(test_lab,num_classes=2)
    train_seq=np.reshape(train_seq,[train_seq.shape[0],train_seq.shape[1],1])
    test_seq=np.reshape(test_seq,[test_seq.shape[0],test_seq.shape[1],1])
    model = kerasNet(int(train_seq.shape[1]),motiflen,'DNA',False,True)
    model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=learning_rate,momentum=momentum_rate,nesterov=True,decay=1e-6,metrics=['binary_accuracy']))
    print('Ready to Fit OK')
    model = to_multi_gpu(model,4)
    #callbacks=[EarlyStopping(monitor='loss', min_delta=math.nan, patience=1)]
    model.fit(train_seq, train_lab,
                  batch_size=batch_size*4,
                  epochs=epochs,# callbacks=callbacks,
                  verbose=1)
    print('Fit complete. Now score and prediction')
    score = model.evaluate(test_seq, test_lab, batch_size=batch_size,verbose=1)
    prediction = model.predict(test_seq,batch_size,verbose=1)
    prediction = np.reshape(prediction,[prediction.shape[0]])
    fpr, tpr, _ = roc_curve(test_lab,prediction)
    roc_auc = auc(fpr, tpr)
    return score,prediction,roc_auc,model

def getNames(listfiles,i):
    train=join(datadir,listfiles[2*i])
    test=join(datadir,listfiles[2*i+1])
    return train,test
            
def save_obj(obj, folder ):
    with open('perf/'+ folder + '/bestauc.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(folder):
    with open('perf/' + folder + '/bestauc.pkl', 'rb') as f:
        return pickle.load(f)

def create_dict(older):    
    diziotot=dict()
    with open(older,'r') as data:
        reader=csv.reader(data, delimiter=',')
        next(reader)
        next(reader)
        for r in reader:
            diziotot[r[1]]=[r[2],r[3],r[4]]
#def create_dict(folder,pad):
#    dictioauc=dict()
#    lista=sorted(listdir(folder))
#    for i in range(0,len(lista),2):
#        if(lista[i][:-pad] in diziotot.keys()):
#            dictioauc[lista[i][:-pad]]=diziotot[lista[i][:-pad]]+[0.]    
    return diziotot

if __name__ == '__main__':
            
            
    
    start_time = time.time()
#    pred = predictPBM('../DREAM5.txt','../DREAM5test.txt','TF_42')
#    print("--- %s seconds ---" % (time.time() - start_time))
#    print(pred)
#    
#    score,auc_score=predictPBMrev('../data/dream5/pbm/sequences.tsv.gz','../data/dream5/pbm/targets.tsv.gz','../data/dream5/chipseq/TF_23_CHIP_51_full_genomic.seq','TF_23')    
#    print('\nscore is ', score)
#    print('\nauc is ',auc_score)
#
#    scoreRNA,predRNA=predictRNA('../data/rnac/sequences.tsv','../data/rnac/targets.tsv','RNCMPT00014',0.8,'RNA')
#    print('score is ', scoreRNA)
#    print('pred is ',predRNA)
    ##################################
    #CHIP
    asktosave=True
#    datadir='../data/encode'    
#    listfiles=sorted(listdir(datadir))
#    for l in range(0,len(listfiles)//2):    
#        training,test=getNames(listfiles,l)
#        exper_name=training[15:-10]
#        print('I am analyzing %s' %(exper_name)) 
#        for i in range(1): 
#            score,pred,auc_score,model=predictChip(training,test)
#            print('\nscore is ',score)
#            #print('\npred is ',pred)
#            print('\nauc is ',auc_score)
#            try:
#                bestaucs=load_obj('chip')
#            except:
#                bestaucs=create_dict('../data/encode',10,'encodeaucs.csv')
#            exper_auc=float(bestaucs[exper_name][2])
#            if(exper_auc<auc_score):
#                if exper_auc!=0:
#                    remove('perf/chip/'+ exper_name + '_model')
#                filename=exper_name + '_model'
#                filepath=join('perf/chip',filename)
#                model.save(filepath)
#                bestaucs[exper_name][2]=auc_score
#                save_obj(bestaucs,'chip')
#            if(asktosave):
#                with open('encodecomplete.csv', 'w') as csv_file:
#                    writer = csv.writer(csv_file)
#                    for key, value in sorted(bestaucs.items()):
#                        writer.writerow([key, float(value[0]),float(value[1]),float(value[2])])
#                    
#        print('best aucs found is ',bestaucs[exper_name])
    #########################################################
        
    ###################################
    #SELEX
    datadir='../data/selex/jolma'    
    listfiles=sorted(listdir(datadir))
    for j in range(0,len(listfiles)//2):
        training,test=getNames(listfiles,j)
        exper_name=training[20:-9]
        print('I am analyzing %s' %(exper_name)) 
        for i in range(3): 
            score,pred,auc_score,model=predictChip(training,test)
            print('\nscore is ',score)
            #print('\npred is ',pred)
            print('\nauc is ',auc_score)
            try:
                bestaucs=load_obj('selex')
            except:
                bestaucs=create_dict('selexauc.csv')
            exper_auc=float(bestaucs[exper_name][2])
            if(exper_auc<auc_score):
                if exper_auc!=0:
                    remove('perf/selex/'+ exper_name + '_model')
                filename=exper_name + '_model'
                filepath=join('perf/selex',filename)
                model.save(filepath)
                bestaucs[exper_name][2]=auc_score
                save_obj(bestaucs,'selex')
            if(asktosave):
                with open('selexcomplete.csv', 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in sorted(bestaucs.items()):
                        writer.writerow([key, float(value[0]),float(value[1]),float(value[2])])
        print('best aucs found is ',bestaucs[exper_name])
#    ########################################################        
#    con model.get_weights posso ottenere i pesi del modello
#    score,pred,auc,model=predictSelex('../data/selex/jolma/Alx1_DBD_TAAAGC20NCG_3_Z_A.seq.gz','../data/selex/jolma/Alx1_DBD_TAAAGC20NCG_3_Z_B.seq.gz')
#    print('\nscore is ',score)
#    print('\npred is ',pred)
#    print('\nauc is ', auc)
    print("--- %s seconds ---" % (time.time() - start_time))    
