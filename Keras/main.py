from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling1D, Conv1D, AveragePooling1D,Input,concatenate
from keras import optimizers
import random
import numpy as np
from util import *
from pbm import *
from rnac import *
from chip import *
from selex import *
#from multi_gpu import *

#inserisci queste due linee all'inizio e alla fine di ogni funzione di cui si vuole verificare il tempo di esecuzione
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))

batch_size = 64 #fissata sempre, su deepfind dovrebbe essere più alta
epochs = 20 #numero di cicli di training, settabile a piacere -> più è alta più dovrebbe essere accurato il training
bases='ACGT'
basesRNA='ACGU'
dictpad={'A':[1.,0.,0.,0.],'C':[0.,1.,0.,0.],'G':[0.,0.,1.,0.],'T':[0.,0.,0.,1.],'U':[0.,0.,0.,1.],'N':[0.25,0.25,0.25,0.25]}
otherArray={'ME':'HK', 'HK':'ME'} #dizionario per ottenere in PBM le sequenze da predire 
dropoutList=[0.5,0.75,1.0] #lista dei possibili valori di dropout
dropoutChoice=random.choice(dropoutList) #estrazione di uno dei valori a caso, dovrebbe essere estratto tramite Bernoulli

#determino learning_rate e momentum_rate estraendo a caso seguendo i due sampler definiti in util
learning_rate=logsampler(0.005,0.05)
momentum_rate=sqrtsampler(0.95,0.99)

#definisco la rete neurale per tutti gli esperimenti
def kerasNet(inpShape,motiflen,exp,hidd):
    inputs=Input(shape=(inpShape,1)) #layer di Input
    conv=Conv1D(filters=16,kernel_size=motiflen*4,activation='relu',strides=4)(inputs) #convoluzione 1 dimensionale con strides=4
    pool=MaxPooling1D(pool_size=int(conv.shape[1]))(conv) #maxpooling
    if exp=='RNA': #se l'esperimento è RNACompete si deve fare concatenazione alternata (ancora da fare) con average
        avgpool=AveragePooling1D(pool_size=int(conv.shape[1]))(conv)
        pool = concatenate([pool,avgpool])#TODO Alternate
    flat=Flatten()(pool) 
    dropout=Dropout(dropoutChoice)(flat) #dropout con la probabilità
    if hidd==True: #hidden layer si o no? migliora le prestazioni?
        dropout=Dense(32, activation='relu',use_bias=True)(dropout) #32 hidden layer con relu
    neu=Dense(1, activation='linear',use_bias=True)(dropout) #passo di rete neurale
    model=Model(inputs=inputs,outputs=neu) #creo il modello collegando l'input all'output
    print (model.summary())
    return model

#################################################################################################################
#PBM    

def predictPBM(fileTrain,fileValid,predTF,exp='DNA'):
    motiflen=24 #motiflen per esperimento PBM
    bestScore=0 #salvo il bestScore e il bestModel per il testing
    bestModel=None 
    bestTf=''
    validSeq,validLab,validStdDev,validAvg,arrayType=createValidation(fileValid,predTF,motiflen) #apre il validation set 
    print('Validation Done')
    dictionary=openPBM(fileTrain,motiflen,True) #apre il training set
    print('I have created the dictionary of %s' %(otherArray[arrayType]))
    listTf=list(dictionary.keys())
    predSeq=predictSequences(fileTrain,arrayType,motiflen) #crea le sequenze da predire
    print('I have collected the predict sequences, entering in the predict cycle...')
    for tfact in listTf: #per ogni fattore di trascrizione in input creo il modello, faccio fitting e valuto la prestazione sul validation
        trainSeq=np.asarray([elem[0] for elem in dictionary[tfact]]) #estraggo solo le sequenze
        trainLab=np.asarray([elem[1] for elem in dictionary[tfact]]) #estratto solo le etichette
        trainLabNorm=(trainLab - np.average(trainLab))/np.std(trainLab) #normalizzo le etichette
        trainSeq=np.reshape(trainSeq,[trainSeq.shape[0],trainSeq.shape[1],1]) #reshape necessario per il training
        print('Training data ready, with TF: %s' %(tfact))        
        
        model = kerasNet(int(trainSeq.shape[1]),motiflen,exp,True)
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
    model = kerasNet(int(train_seq.shape[1]),motiflen,exp,True)
    print('Construction ok Keras network OK')
    model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=0.0001,momentum=0.95,nesterov=True,decay=1e-6),
                      metrics=['mae'])
    print('Ready to Fit OK')
    model.fit(train_seq, train_lab_norm,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(valid_seq, valid_lab_norm))
    print('Fit complete. Now score and prediction')
    if invivo!='': #faccio testing su dati in vivo
        prediction_res=[]
        test_seq2=testRnaVivo(invivo,16,41)
        for testseq in test_seq2:
            test_seq2=np.reshape(test_seq2,[test_seq2.shape[0],test_seq2.shape[1],1])
            prediction = model.predict(test_seq2,batch_size,verbose=1)
            prediction_res.append(prediction)
        return prediction_res
        
    score = model.evaluate(test_seq, test_lab, verbose=1)
    prediction = model.predict(test_seq,batch_size,verbose=1)
    prediction = prediction*np.std(test_lab) + np.average(test_lab)
    prediction = np.reshape(prediction,[prediction.shape[0]]) 
    #statsRNA(orig_test_dataset,prediction,test_lab)
    return score,prediction
        
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
#    train_seq=np.reshape(train_seq,[train_seq.shape[0],train_seq.shape[1],1])
#    train_seq=np.reshape(train_seq,[train_seq.shape[0],train_seq.shape[1],1])
    model = kerasNet(int(train_seq.shape[1]),motiflen,'DNA',False)
    #model= make_parallel(model,4)    
    model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=learning_rate,momentum=momentum_rate,nesterov=True,decay=1e-6),
                      metrics=['mae'])
    print('Ready to Fit OK') 
    model.fit(train_seq, train_lab,
                  batch_size=batch_size*4,
                  epochs=epochs,
                  verbose=1)
    print('Fit complete. Now score and prediction')
    
    score = model.evaluate(test_seq, test_lab, batch_size=batch_size,verbose=1)
    prediction = model.predict(test_seq,batch_size,verbose=1)
    prediction = np.reshape(prediction,[prediction.shape[0]]) 
    #statsRNA(orig_test_dataset,prediction,test_lab)
    return score,prediction
    
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
#    train_seq=np.reshape(train_seq,[train_seq.shape[0],train_seq.shape[1],1])
#    train_seq=np.reshape(train_seq,[train_seq.shape[0],train_seq.shape[1],1])
    model = kerasNet(int(train_seq.shape[1]),motiflen,'DNA',True)
    model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=learning_rate,momentum=momentum_rate,nesterov=True,decay=1e-6),
                      metrics=['mae'])
    print('Ready to Fit OK')
    model.fit(train_seq, train_lab,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1)
    print('Fit complete. Now score and prediction')
    score = model.evaluate(test_seq, test_lab, batch_size=batch_size,verbose=1)
    prediction = model.predict(test_seq,batch_size,verbose=1)
    prediction = np.reshape(prediction,[prediction.shape[0]]) 
    #statsRNA(orig_test_dataset,prediction,test_lab)
    return score,prediction

if __name__ == '__main__':
    pred = predictPBM('../DREAM5.txt','../DREAM5test.txt','TF_42')
    print(pred)
#   scoreRNA,predRNA=predictRNA('../data/rnac/sequences.tsv','../data/rnac/targets.tsv','RNCMPT00014',0.8,'RNA')
#   print('score is ', scoreRNA)
#    score,pred=predictChip('../data/encode/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_AC.seq.gz','../data/encode/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_B.seq.gz')
#    print('\nscore is ',score)
#    print('pred is ',pred)
#    score,pred=predictSelex('../data/selex/jolma/Alx1_DBD_TAAAGC20NCG_3_Z_A.seq.gz','../data/selex/jolma/Alx1_DBD_TAAAGC20NCG_3_Z_B.seq.gz')
#    print('\nscore is ',score)
#    print('pred is ',pred)
