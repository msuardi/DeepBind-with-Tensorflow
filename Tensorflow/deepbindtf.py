import tensorflow as tf
import numpy as np
import csv
import math 
import random
import gzip
from scipy.stats import bernoulli
import zipfile as zp

nummotif=16 #number of motifs to discover
bases='ACGT' #DNA bases
basesRNA='ACGU'#RNA bases
batch_size=64 #fixed batch size -> see notes to problem about it
dictReverse={'A':'T','C':'G','G':'C','T':'A','N':'N'} #dictionary to implement reverse-complement mode
 
 
#function that produce a tensor made by the concatenate in alternate fashion of two tensors
def altconcat(a,b,batchsize,kind='normal'):
    conc=tf.concat([a,b],1)
    conc=tf.reshape(conc,[batchsize,2,nummotif])
    result=[]
    for i in range(0,int(conc.shape[0])):
        for j in range(0,int(conc.shape[2])):
            result=tf.concat([result,conc[i][:,j]],0)
    return result

#function that produce a real number in a range [0,1), supp 2.1 following log sampler
def logsampler(a,b,tensor=0):
    if(tensor==1):
        x=tf.Variable(tf.random_uniform([1],minval=0,maxval=1))
    else:
        x=np.random.uniform(low=0,high=1)
    y=10**((math.log10(b)-math.log10(a))*x + math.log10(a))
    return y

#function that produce a real number in a range [0,1) supp 2.1 following sqrt sampler
def sqrtsampler(a,b,tensor=0):
    if(tensor==1):
        x=tf.Variable(tf.random_uniform([1],minval=0,maxval=1))
    else:
        x=np.random.uniform(low=0,high=1)
    y=(b-a)*math.sqrt(x)+a
    return y

#function that converts a sequence to a padded one-of-four representation
def seqtopad(sequence,motlen,kind='DNA'):
    rows=len(sequence)+2*motlen-2
    S=np.empty([rows,4])
    base= bases if kind=='DNA' else basesRNA
    for i in range(rows):
        for j in range(4):
            if i-motlen+1<len(sequence) and sequence[i-motlen+1]=='N' or i<motlen-1 or i>len(sequence)+motlen-2:
                S[i,j]=np.float32(0.25)
            elif sequence[i-motlen+1]==base[j]:
                S[i,j]=np.float32(1)
            else:
                S[i,j]=np.float32(0)
    return S

#function to create batch of size<64, necessary without padding of sequences (returns size if batch) --> NOW NOT USED
def nextbatch(dataset, index):
    if len(dataset[index+64][0])==len(dataset[index][0]):
        return int(64)
    else:
        for i in range(64):
            if len(dataset[index+i][0])!=len(dataset[index][0]):
                return int(i)
                
#function that returns the opposite strand of the sequence
def reverse(sequence):
    revseq=''
    for i in sequence:
        revseq+=dictReverse[i]
    return revseq
        
#function that implements a dinucshuffle
def dinucshuffle(sequence):
    b=[sequence[i:i+2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d=''.join([str(x) for x in b])
    return d

#convolution operation
def convolution(weights, inp, motif):
    first=tf.reshape(weights[0],[motiflen,4,1,1])
    conv=tf.nn.conv2d(inp,first,strides=[1,1,1,1],padding='VALID')
    for i in range(1,len(weights)):
        other=tf.reshape(weights[i],[motif,4,1,1])
        iconv=tf.nn.conv2d(inp,other,strides=[1,1,1,1],padding='VALID')
        conv=tf.concat([conv,iconv],0)
    return conv
    
class Experiment:
    def __init__(self,filename,motiflen):
        self.file=filename
        self.motiflen=motiflen
    
    def getMotifLen(self):
        return self.motiflen

def checkSelex(filename):
    filesplit=filename.split('_')[2]
    numb=[s for s in filesplit if s.isdigit()]
    nu=''.join([str(x) for x in numb])
    return int(nu)

class Selex(Experiment):
    def __init__(self,filename):
        self.file=filename
        self.motiflen=checkSelex(filename)
    def openFile(self):
        train_dataset=[]
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader=csv.reader(data,delimiter='\t')
            for row in reader:
                    train_dataset.append([seqtopad(row[2],self.motiflen),[int(row[3])]])
        random.shuffle(train_dataset)
        frac1=int(len(train_dataset)*0.8)
        frac2=int(len(train_dataset)*0.9)
        return train_dataset[:frac1],train_dataset[frac1:frac2],train_dataset[frac2:]    


class Chip(Experiment):
    def __init__(self,filename,motiflen=24):
        self.file = filename
        self.motiflen = motiflen
            
    def openFile(self):
        train_dataset=[]
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            for row in reader:
                    train_dataset.append([seqtopad(row[2],self.motiflen),[1]])
                    train_dataset.append([seqtopad(dinucshuffle(row[2]),self.motiflen),[0]])
        random.shuffle(train_dataset)
        frac1=int(len(train_dataset)*0.8)
        frac2=int(len(train_dataset)*0.9)
        return train_dataset[:frac1],train_dataset[frac1:frac2],train_dataset[frac2:]

        
class Pbm(Experiment):
    def __init__(self,sequencefile,targetfile,motiflen=24):
        self.sequence=sequencefile
        self.target=targetfile
        self.motiflen=24
    def openFile(self,tfactor,reverseMode):
        train_dataset=[]
        train_labels=[]
        clearLabel=[]
        train_dataset_pad=[]
    
        #open target file and save the indices of nan entries
        with gzip.open(self.target, 'rt') as data:
            next(data)
            reader=csv.reader(data,delimiter='\t')
            for row in reader:
                if math.isnan(np.float(row[tfactor]))==False:            
                    train_labels.append(np.float32(row[tfactor]))
                else:
                    clearLabel.append(reader.line_num)
       
       #open sequence file and save sequences with the specificity value
        with gzip.open(self.sequence, 'rt') as data:
            next(data)
            reader=csv.reader(data,delimiter='\t')
            for row in reader:
                if reader.line_num not in clearLabel:
                    train_dataset.append([[row[2]],[train_labels[len(train_dataset)]]])
        #sort dataset by the length of the sequence
        train_dataset.sort(key=lambda s: len(s[0][0]))
       
       #transform sequence to padded representation, adding the opposite strand
        train_dataset_pad=[]
        if(reverseMode):
            for i in range(len(train_dataset)):
                train_dataset_pad.append([seqtopad(train_dataset[i][0][0],self.motiflen),train_dataset[i][1]])
                train_dataset_pad.append([seqtopad(reverse(train_dataset[i][0][0]),self.motiflen),train_dataset[i][1]])
        else:
            for i in range(len(train_dataset)):
                train_dataset_pad.append([seqtopad(train_dataset[i][0][0],self.motiflen),train_dataset[i][1]])
        random.shuffle(train_dataset_pad)
#        
#        frac1=int(len(train_dataset_pad)*0.8)
#        frac2=int(len(train_dataset_pad)*0.9)
#        return train_dataset_pad[:frac1],train_dataset_pad[frac1:frac2],train_dataset_pad[frac2:]
        size=int(len(train_dataset_pad)/3)
        start1=random.choice(range(len(train_dataset_pad)-size))
        start2=random.choice(range(len(train_dataset_pad)-size))
        start3=random.choice(range(len(train_dataset_pad)-size))
        firsttrain=train_dataset_pad[:start1]+train_dataset_pad[start1+size:]
        secondtrain=train_dataset_pad[:start2]+train_dataset_pad[start2+size:]
        thirdtrain=train_dataset_pad[:start3]+train_dataset_pad[start3+size:]
        firstvalid=train_dataset_pad[start1:start1+size]
        secondvalid=train_dataset_pad[start2:start2+size]
        thirdvalid=train_dataset_pad[start3:start3+size]
        
        return firsttrain,firstvalid,secondtrain,secondvalid,thirdtrain,thirdvalid,train_dataset_pad
        
class Pbm2(Experiment):
    def __init__(self,sequencefile,motiflen=24):
        self.sequence=sequencefile
        self.motiflen=24
    def openFile(self,reverseMode):
        train_dataset=[]
        train_dataset_pad=[]
    
        #open target file and save the indices of nan entries
        with open(self.sequence,'rt') as data:
            next(data)
            reader=csv.reader(data,delimiter='\t')
            for row in reader:
                if reader.line_num <80000:
                    if math.isnan(np.float(row[3]))==False:            
                        train_dataset.append([seqtopad(row[2],self.motiflen),[np.float(row[3])]])
       
       #transform sequence to padded representation, adding the opposite strand
        train_dataset_pad=train_dataset
#        if(reverseMode):
#            for i in range(len(train_dataset)):
#                train_dataset_pad.append([seqtopad(train_dataset[i][0][0],self.motiflen),train_dataset[i][1]])
#                train_dataset_pad.append([seqtopad(reverse(train_dataset[i][0][0]),self.motiflen),train_dataset[i][1]])
#        else:
#            for i in range(len(train_dataset)):
#                train_dataset_pad.append([seqtopad(train_dataset[i][0][0],self.motiflen),train_dataset[i][1]])
        random.shuffle(train_dataset_pad)
#        
#        frac1=int(len(train_dataset_pad)*0.8)
#        frac2=int(len(train_dataset_pad)*0.9)
#        return train_dataset_pad[:frac1],train_dataset_pad[frac1:frac2],train_dataset_pad[frac2:]
        train_lab = [el[1][0] for el in train_dataset_pad]
        train_lab2 = (train_lab - np.average(train_lab))/np.std(train_lab)
        train_dataset_pad=[[train_dataset_pad[i][0],train_lab2[i]] for i in len(train_lab2)]
        size=int(len(train_dataset_pad)/3)
        start1=random.choice(range(len(train_dataset_pad)-size))
        start2=random.choice(range(len(train_dataset_pad)-size))
        start3=random.choice(range(len(train_dataset_pad)-size))
        firsttrain=train_dataset_pad[:start1]+train_dataset_pad[start1+size:]
        secondtrain=train_dataset_pad[:start2]+train_dataset_pad[start2+size:]
        thirdtrain=train_dataset_pad[:start3]+train_dataset_pad[start3+size:]
        firstvalid=train_dataset_pad[start1:start1+size]
        secondvalid=train_dataset_pad[start2:start2+size]
        thirdvalid=train_dataset_pad[start3:start3+size]
        
        return firsttrain,firstvalid,secondtrain,secondvalid,thirdtrain,thirdvalid,train_dataset_pad
        
class Rnac(Experiment):
    def __init__(self,sequencefile,targetfile,motiflen=16):
        self.sequence = sequencefile
        self.target = targetfile
        self.motiflen = 16
    def openFile(self,tfactor):
        train_dataset=[]
        train_labels=[]
        clearLabel=[]
        train_dataset_pad=[]
    
        #open target file and save the indices of nan entries
        with open(self.target,'r') as data:
            next(data)
            reader=csv.reader(data,delimiter='\t')
            for row in reader:
                if math.isnan(np.float(row[tfactor]))==False:            
                    train_labels.append(np.float32(row[tfactor]))
                else:
                    clearLabel.append(reader.line_num)
       
       #open sequence file and save sequences with the specificity value
        with open(self.sequence,'r') as data:
            next(data)
            reader=csv.reader(data,delimiter='\t')
            for row in reader:
                if reader.line_num not in clearLabel:
                    train_dataset.append([[row[2]],[train_labels[len(train_dataset)]]])
        #sort dataset by the length of the sequence
        train_dataset.sort(key=lambda s: len(s[0][0]))
        
        #pad sequences in RNAcompete, different length
        if(len(train_dataset[0][0][0])!=len(train_dataset[len(train_dataset)-1][0][0])):
            for i in range(len(train_dataset)):
                lunmax=len(train_dataset[len(train_dataset)-1][0][0]) 
                if(len(train_dataset[i][0][0])<lunmax):
                    train_dataset[i][0][0]=train_dataset[i][0][0]+'N'*(lunmax-len(train_dataset[i][0][0]))
       
       #transform sequence to padded representation, adding the opposite strand
        train_dataset_pad=[]
        for i in range(len(train_dataset)):
            train_dataset_pad.append([seqtopad(train_dataset[i][0][0],self.motiflen,'RNA'),train_dataset[i][1]])
        random.shuffle(train_dataset_pad)
        frac1=int(len(train_dataset_pad)*0.8)
        frac2=int(len(train_dataset_pad)*0.9)
        return train_dataset_pad[:frac1],train_dataset_pad[frac1:frac2],train_dataset_pad[frac2:]
        


############################################################
#NEURAL NETWORK BATCH VERSION

pbmPred = Pbm2('/home/mauro/Downloads/DREAM5_PBM_Data_TrainingSet.txt')
train1,valid1,train2,valid2,train3,valid3,alldataset = pbmPred.openFile(False)
train_dataset=[train1,train2,train3]
valid_dataset=[valid1,valid2,valid3]
motiflen = pbmPred.getMotifLen()
valid_dataset_tot=[]
valid_labels_tot=[]
for i in range(3):
    valid_data=np.asarray([el[0] for el in valid_dataset[i]],dtype=np.float32)
    valid_lab=np.asarray([el[1] for el in valid_dataset[i]],dtype=np.float32)
    valid_dataset_tot.append(valid_data)
    valid_labels_tot.append(valid_lab)





#rnacPred= Rnac('data/rnac/sequences.tsv','data/rnac/targets.tsv')
#train_dataset2,valid_dataset_tot2,test_dataset_tot2=rnacPred.openFile(12)
#motiflen2 = rnacPred.getMotifLen()

#chipPred = Chip('data/encode/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_AC.seq.gz')
#train_dataset,valid_dataset_tot,test_dataset_tot=chipPred.openFile()
#motiflen = chipPred.getMotifLen()

#selexPred = Selex('data/selex/jolma/Alx1_DBD_TAAAGC20NCG_3_Z_B.seq.gz')
#train_dataset,valid_dataset_tot,test_dataset_tot=selexPred.openFile()
#motiflen = selexPred.getMotifLen()


#valid_dataset=np.asarray([el[0] for el in valid_dataset_tot],dtype=np.float32)
#valid_labels=np.asarray([el[1] for el in valid_dataset_tot],dtype=np.float32)
#test_dataset=np.asarray([el[0] for el in test_dataset_tot],dtype=np.float32)
#test_labels=np.asarray([el[1][0] for el in test_dataset_tot],dtype=np.float32)

#parameters for training




graph=tf.Graph()
with graph.as_default():
    labelNetwork=tf.placeholder(tf.float32, [batch_size,1],name='labelNetwork') #label of each input is the sensitivity
    inputNetworkBatch=tf.placeholder(tf.float32,[batch_size, None ,4],name='inputNetworkBatch') #placeholder will receive input (64x70x4 is output size of seqtopad for sequence of length 40)
    #inputValidBatch=tf.placeholder(tf.float32,[len(valid_dataset_tot[0]), len(valid_dataset_tot[0][0]) ,4],name='inputValidBatch')        
    inputValid1=tf.constant(valid_dataset_tot[0],shape=[len(valid_dataset_tot[0]), len(valid_dataset_tot[0][0]) ,4],name='inputValid1')
    inputValid2=tf.constant(valid_dataset_tot[1],shape=[len(valid_dataset_tot[1]), len(valid_dataset_tot[1][0]) ,4],name='inputValid2')
    inputValid3=tf.constant(valid_dataset_tot[2],shape=[len(valid_dataset_tot[2]), len(valid_dataset_tot[2][0]) ,4],name='inputValid3')
    #tf_valid_dataset = tf.constant(valid_dataset, shape=[len(valid_dataset),len(valid_dataset[0]),4])
   # tf_test_dataset = tf.constant(test_dataset,shape=[len(test_dataset),len(valid_dataset[0]),4])
    #standard deviation for convolution stage    
    sigmaConv=logsampler(10**-7,10**-3) 
    #weights for convolution and rectification stage
    wConv=tf.Variable(tf.truncated_normal([nummotif,motiflen,4],mean=0,stddev=sigmaConv))#tunable weights for convolution stage (16x16x4 is size of tunable weights ) 
    wRect=tf.Variable(tf.truncated_normal([nummotif])) #tunable weights for rectification stage
    #what is the distribution of of the rectification?
    sigmaNeu=logsampler(10**-5,10**-2)     #standard deviation for neural network stage
    wNeu=tf.Variable(tf.truncated_normal([nummotif,1],mean=0,stddev=sigmaNeu)) #tunable weights for neural network stage
    wNeuMaxAvg=tf.Variable(tf.truncated_normal([2*nummotif,1],mean=0,stddev=sigmaNeu)) #tunable weights for neural network stage with MaxAvg pooling
    wNeuMaxHidden=tf.Variable(tf.truncated_normal([2*nummotif,1],mean=0,stddev=sigmaNeu)) #tunable weights for neural network stage with Max and Hidden
    wNeuBias=tf.Variable(tf.truncated_normal([1],mean=0,stddev=sigmaNeu)) #tunable bias for neural network stage 
    
    wHidden=tf.Variable(tf.truncated_normal([32,nummotif],mean=0,stddev=sigmaNeu)) #hidden weights with max pooling
    wHiddenMaxAvg=tf.Variable(tf.truncated_normal([32,2*nummotif],mean=0,stddev=sigmaNeu))#hidden weights for neural network with MaxAvg pooling
    wHiddenBias=tf.Variable(tf.truncated_normal([32,1],mean=0,stddev=sigmaNeu)) #hidden bias for everything
    
    #DROPOUT VARIABLES
    #Dropout for no hidden models
    dropoutList=[0.5,0.75,1.0] #list of possible dropout values
    dropoutChoice=random.choice(dropoutList) #choose random in list, it should be extract with Bernoulli
    
    #beta values for the training objective
    beta1=logsampler(10**-10,10**-3,1)
    beta2=logsampler(10**-15,10**-3,1)
    beta3=logsampler(10**-15,10**-3,1)
    
        
    #create the neural net
    def conv_net(inp, batch_size, wconvolution,wrect, wneural, wneuralbias,whidd,whiddbi,dropprob,poolType='max',neuType='nohidden',dataset='training'):
        # Convolution Stage
        inp = tf.reshape(inp, shape=[batch_size,-1,4,1]) #reshape input        
        unstack=tf.unstack(wconvolution)
        conv = convolution(unstack,inp,motiflen)
        print(conv.shape, 'convolution')
        conv=tf.squeeze(conv)
        conv=tf.reshape(conv,shape=[batch_size,-1,nummotif])
        print(conv.shape, 'reshapandolo')
        #Rectification Stage
        convPost= tf.subtract(conv,wrect) #subtract weights
        rect=tf.nn.relu(convPost) #do rectification
        # Reshape size
        rect=tf.reshape(rect,shape=[batch_size,-1,nummotif,1]) #reshape rectification    
        # Pool Stage, I use tf.reduce_max/mean instead of tf.nn.max/avg_pool because the last one doesn't support dynamic ksize
        maxpool=tf.reduce_max(rect,axis=1) #max pool in every case
        #check if it needs a maxavg pool
        if(poolType=='maxavg'):
            avgpool=tf.reduce_mean(rect,axis=1) #avg pool        
            pool=altconcat(maxpool,avgpool,batch_size,'batch') #concatenate with alternate fashion
            pool=tf.reshape(pool,shape=[batch_size,2*nummotif]) #reshape pool, it's 2d-dimensional
        else:
            pool=tf.reshape(maxpool,shape=[batch_size,nummotif]) #reshape pool, it's d-dimensional
        # Neural Network Stage
        mask=bernoulli(dropprob).rvs(size=pool.get_shape().as_list())
        #check if there's hidden stage
        if(neuType=='nohidden'):
            #pooldrop=tf.nn.dropout(pool,keep_prob=mask)
            if dataset=='training':
                pooldrop=tf.multiply(pool,mask)            
                pooldrop=tf.reshape(pooldrop,[batch_size,int(pooldrop.shape[1])])
                out=tf.add(tf.matmul(pooldrop,wneural),wneuralbias) #neural stage
            else:
                out=tf.add(dropoutChoice*tf.matmul(pool,wneural),wneuralbias)
        else:
            mult=tf.add(tf.matmul(pool,tf.reshape(whidd[0],shape=[int(whidd[0].shape[0]),1])),whiddbi[0]) #if hidden, first element
            outtemp=tf.nn.relu(mult) #rectify this 
            for j in range(1,32):
                mult=tf.add(tf.matmul(pool,tf.reshape(whidd[j],shape=[int(whidd[0].shape[0]),1])),whiddbi[j]) #other elements
                rectmult=tf.nn.relu(mult) #rectify
                outtemp=tf.concat([outtemp,rectmult],1) #concatenate results
            #pooldrop=tf.nn.dropout(outtemp,keep_prob=mask)
            if dataset=='training':
                pooldrop=tf.multiply(pool,mask)            
                pooldrop=tf.reshape(pooldrop,[batch_size,int(pooldrop.shape[1])])
                out=tf.add(tf.matmul(outtemp,wneural),wneuralbias) #neural final stage
                #out=tf.add(tf.reduce_sum(tf.multiply(outtemp,tf.squeeze(wneural)),1),wneuralbias) #it works
            else:
                out=tf.add(dropoutChoice*tf.matmul(outtemp,wneural),wneuralbias)
        
        return out #single value, predict specificity
       
    ########################################################################################################################################       
    
    
    #NEURAL NETWORK CHANCES
    #MAXHIDDEN
    pred = conv_net(inputNetworkBatch, batch_size, wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden')
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=labelNetwork))
    #loss=tf.reduce_sum(tf.losses.mean_squared_error(labelNetwork,pred))#+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHidden))+beta3*tf.reduce_sum(tf.abs(wNeuMaxHidden))
    #loss=tf.reduce_mean(-1*tf.losses.log_loss(pred,labelNetwork))+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHidden))+beta3*tf.reduce_sum(tf.abs(wNeuMaxHidden))
    loss=tf.reduce_mean(tf.subtract(labelNetwork,pred)**2 / (2*batch_size))#+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHidden))+beta3*tf.reduce_sum(tf.abs(wNeuMaxHidden))
    ##MAXNOHIDDEN
    #pred = conv_net(inputNetworkBatch, batch_size, wConv,wRect, wNeu, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','nohidden')
    #loss=tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred,labels=labelNetwork)+beta1*tf.reduce_sum(tf.abs(wConv))+beta3*tf.reduce_sum(tf.abs(wNeu)))
    #loss = tf.reduce_mean(0.5*tf.pow(tf.subtract(pred,labelNetwork),2))+beta1*tf.reduce_sum(tf.abs(wConv))+beta3*tf.reduce_sum(tf.abs(wNeu))
    
    
    #JUST FOR RNACompete    
    ##MAXAVGHIDDEN
    #pred = conv_net(inputNetworkBatch, batch_size,wConv,wRect, wNeuMaxAvg, wNeuBias,wHiddenMaxAvg,wHiddenBias,dropoutChoice,'maxavg','hidden')
    #loss=tf.reduce_mean(tf.losses.mean_squared_error(labelNetwork,pred))+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHiddenMaxAvg))+beta3*tf.reduce_sum(tf.abs(wNeuMaxAvg))
    #loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(pred,labelNetwork)))+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHiddenMaxAvg))+beta3*tf.reduce_sum(tf.abs(wNeuMaxAvg))
    ##MAXAVGNOHIDDEN
    #pred = conv_net(inputNetworkBatch, batch_size,wConv,wRect, wNeuMaxAvg, wNeuBias,wHiddenMaxAvg,wHiddenBias,dropoutChoice,'maxavg','nohidden')
    #loss=tf.reduce_mean(tf.losses.mean_squared_error(labelNetwork,pred))+beta1*tf.reduce_sum(tf.abs(wConv))+beta3*tf.reduce_sum(tf.abs(wNeuMaxAvg))
        
    #########################################################################################################################################    
    learning_rate=logsampler(0.005,0.05)
    momentum_rate=sqrtsampler(0.95,0.99)    


    #optimize with gradient descent with momentum, nesterov applied (paper say this)    
    #optimizer=tf.train.MomentumOptimizer(learning_rate,momentum_rate,use_nesterov=True).minimize(loss)
    ##optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
     #train_prediction = tf.nn.softmax(pred)
    
    accuracy=tf.reduce_mean(tf.abs(tf.subtract(pred,labelNetwork)))
    valid_prediction1=conv_net(inputValid1,len(valid_dataset_tot[0]), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','validation')
    valid_prediction2=conv_net(inputValid2,len(valid_dataset_tot[0]), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','validation')
    valid_prediction3=conv_net(inputValid3,len(valid_dataset_tot[0]), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','validation')
    valid_prediction=[valid_prediction1,valid_prediction2,valid_prediction3]

   # test_prediction=conv_net(tf_test_dataset, len(test_dataset), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','test')
    #testlab=tf.constant(test_labels,shape=[len(test_labels),1])
    #accurval=tf.reduce_mean(tf.abs(tf.subtract(tf.squeeze(test_prediction),labelTest)))

        #i did it, accuracy maybe it's the mean of this values
    
        #maybe I need these lines for SELEX and CHIP-seq
        #correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #def accuracy(predictions, labels):
        #  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    
    #def accuracy(predictions, labels):
    #    sum=0    
    #    for i in range(len(predictions)):
    #        if(predictions[i]==labels[i]):
    #            sum+=1
    #    return (100.0 * sum / len(predictions))

resa=[]
out=[]

with tf.Session(graph=graph) as sess:
    best=0
    for i in range(2):
        sess.run(tf.global_variables_initializer()) #initialize variables
        print('Initialized all Variables',i)
        lastphase=0
        termin=batch_size
        start=0
        evaluation=[]
        for count in range(3):
            while termin  < len(train_dataset[0]):
                batch_data=[el[0] for el in train_dataset[count][start:(start + batch_size)]] 
                batch_labels=[el[1] for el in train_dataset[count][start:(start + batch_size)]]
                start=termin
                termin=termin+batch_size
                feeddict={inputNetworkBatch: batch_data, labelNetwork: batch_labels}
                sess.run(optimizer,feed_dict=feeddict)
                predictions,l,_,acc= sess.run([pred,loss,optimizer,accuracy],feed_dict=feeddict)
                resa.append(predictions)
                #out.append(wConv[0].eval())
                phase = int((termin)/len(train_dataset[0])*100) 
                if (phase % 5 ==0) and (phase!=lastphase):
                   #print("Minibatch loss at step %d %% : %.3f" % (phase, l))
                   lastphase=phase
                   print("Phase: %d %% \nAccuracy is: %f" % (phase,acc))
                   #accval=np.average(np.abs(np.subtract(valid_prediction.eval(),valid_labels)))
            accval=np.average(np.abs(np.subtract(valid_prediction[count].eval(),valid_labels_tot[count])))
            evaluation.append(accval)
        print(np.average(evaluation))
        
        #why these values are the first of the cycle and not the last one? which I have to save?
        if best==0 or best>np.average(evaluation):
            best=np.average(evaluation)
            saveLearningRate=learning_rate
            saveLearningMomentum=momentum_rate
            saveWConv=wConv.eval()
            saveWNeuMaxHidden=wNeuMaxHidden.eval()
            saveNeuBias=wNeuBias.eval()
            saveWHidden=wHidden.eval()
            saveWHiddenBias=wHiddenBias.eval()
            
            
graph2=tf.Graph()
with graph2.as_default():
    labelNetwork=tf.placeholder(tf.float32, [batch_size,1],name='labelNetwork') #label of each input is the sensitivity
    inputNetworkBatch=tf.placeholder(tf.float32,[batch_size, None ,4],name='inputNetworkBatch') #placeholder will receive input (64x70x4 is output size of seqtopad for sequence of length 40)
    #standard deviation for convolution stage    
    #sigmaConv=logsampler(10**-7,10**-3) 
    #weights for convolution and rectification stage
    wConv=tf.Variable(saveWConv)#tunable weights for convolution stage (16x16x4 is size of tunable weights ) 
    wRect=tf.Variable(tf.truncated_normal([nummotif])) #tunable weights for rectification stage
    #what is the distribution of of the rectification?
    sigmaNeu=logsampler(10**-5,10**-2)     #standard deviation for neural network stage
#    wNeu=tf.Variable(tf.truncated_normal([nummotif,1],mean=0,stddev=sigmaNeu)) #tunable weights for neural network stage
#    wNeuMaxAvg=tf.Variable(tf.truncated_normal([2*nummotif,1],mean=0,stddev=sigmaNeu)) #tunable weights for neural network stage with MaxAvg pooling
    wNeuMaxHidden=tf.Variable(saveWNeuMaxHidden) #tunable weights for neural network stage with Max and Hidden
    wNeuBias=tf.Variable(saveNeuBias) #tunable bias for neural network stage 
    
    wHidden=tf.Variable(saveWHidden) #hidden weights with max pooling
#    wHiddenMaxAvg=tf.Variable(tf.truncated_normal([32,2*nummotif],mean=0,stddev=sigmaNeu))#hidden weights for neural network with MaxAvg pooling
    wHiddenBias=tf.Variable(saveWHiddenBias) #hidden bias for everything
    
    #DROPOUT VARIABLES
    #Dropout for no hidden models
    dropoutList=[0.5,0.75,1.0] #list of possible dropout values
    dropoutChoice=random.choice(dropoutList) #choose random in list, it should be extract with Bernoulli
    
    #beta values for the training objective
    beta1=logsampler(10**-10,10**-3,1)
    beta2=logsampler(10**-15,10**-3,1)
    beta3=logsampler(10**-15,10**-3,1)
    
        
    #create the neural net
    def conv_net(inp, batch_size, wconvolution,wrect, wneural, wneuralbias,whidd,whiddbi,dropprob,poolType='max',neuType='nohidden',dataset='training'):
        # Convolution Stage
        inp = tf.reshape(inp, shape=[batch_size,-1,4,1]) #reshape input        
        unstack=tf.unstack(wconvolution)
        conv = convolution(unstack,inp,motiflen)
        print(conv.shape, 'convolution')
        conv=tf.squeeze(conv)
        conv=tf.reshape(conv,shape=[batch_size,-1,nummotif])
        print(conv.shape, 'reshapandolo')
        #Rectification Stage
        convPost= tf.subtract(conv,wrect) #subtract weights
        rect=tf.nn.relu(convPost) #do rectification
        # Reshape size
        rect=tf.reshape(rect,shape=[batch_size,-1,nummotif,1]) #reshape rectification    
        # Pool Stage, I use tf.reduce_max/mean instead of tf.nn.max/avg_pool because the last one doesn't support dynamic ksize
        maxpool=tf.reduce_max(rect,axis=1) #max pool in every case
        #check if it needs a maxavg pool
        if(poolType=='maxavg'):
            avgpool=tf.reduce_mean(rect,axis=1) #avg pool        
            pool=altconcat(maxpool,avgpool,batch_size,'batch') #concatenate with alternate fashion
            pool=tf.reshape(pool,shape=[batch_size,2*nummotif]) #reshape pool, it's 2d-dimensional
        else:
            pool=tf.reshape(maxpool,shape=[batch_size,nummotif]) #reshape pool, it's d-dimensional
        # Neural Network Stage
        mask=bernoulli(dropprob).rvs(size=pool.get_shape().as_list())
        #check if there's hidden stage
        if(neuType=='nohidden'):
            #pooldrop=tf.nn.dropout(pool,keep_prob=mask)
            if dataset=='training':
                pooldrop=tf.multiply(pool,mask)            
                pooldrop=tf.reshape(pooldrop,[batch_size,int(pooldrop.shape[1])])
                out=tf.add(tf.matmul(pooldrop,wneural),wneuralbias) #neural stage
            else:
                out=tf.add(dropoutChoice*tf.matmul(pool,wneural),wneuralbias)
        else:
            mult=tf.add(tf.matmul(pool,tf.reshape(whidd[0],shape=[int(whidd[0].shape[0]),1])),whiddbi[0]) #if hidden, first element
            outtemp=tf.nn.relu(mult) #rectify this 
            for j in range(1,32):
                mult=tf.add(tf.matmul(pool,tf.reshape(whidd[j],shape=[int(whidd[0].shape[0]),1])),whiddbi[j]) #other elements
                rectmult=tf.nn.relu(mult) #rectify
                outtemp=tf.concat([outtemp,rectmult],1) #concatenate results
            #pooldrop=tf.nn.dropout(outtemp,keep_prob=mask)
            if dataset=='training':
                pooldrop=tf.multiply(pool,mask)            
                pooldrop=tf.reshape(pooldrop,[batch_size,int(pooldrop.shape[1])])
                out=tf.add(tf.matmul(outtemp,wneural),wneuralbias) #neural final stage
                #out=tf.add(tf.reduce_sum(tf.multiply(outtemp,tf.squeeze(wneural)),1),wneuralbias) #it works
            else:
                out=tf.add(dropoutChoice*tf.matmul(outtemp,wneural),wneuralbias)
        
        return out #single value, predict specificity
       
    ########################################################################################################################################       
    
    
    #NEURAL NETWORK CHANCES
    #MAXHIDDEN
    pred = conv_net(inputNetworkBatch, batch_size, wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden')
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=labelNetwork))
    #loss=tf.reduce_sum(tf.losses.mean_squared_error(labelNetwork,pred))#+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHidden))+beta3*tf.reduce_sum(tf.abs(wNeuMaxHidden))
    #loss=tf.reduce_mean(-1*tf.losses.log_loss(pred,labelNetwork))+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHidden))+beta3*tf.reduce_sum(tf.abs(wNeuMaxHidden))
    loss=tf.reduce_mean(tf.subtract(labelNetwork,pred)**2 / (2*batch_size))#+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHidden))+beta3*tf.reduce_sum(tf.abs(wNeuMaxHidden))
    ##MAXNOHIDDEN
    #pred = conv_net(inputNetworkBatch, batch_size, wConv,wRect, wNeu, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','nohidden')
    #loss=tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred,labels=labelNetwork)+beta1*tf.reduce_sum(tf.abs(wConv))+beta3*tf.reduce_sum(tf.abs(wNeu)))
    #loss = tf.reduce_mean(0.5*tf.pow(tf.subtract(pred,labelNetwork),2))+beta1*tf.reduce_sum(tf.abs(wConv))+beta3*tf.reduce_sum(tf.abs(wNeu))
    
    
    #JUST FOR RNACompete    
    ##MAXAVGHIDDEN
    #pred = conv_net(inputNetworkBatch, batch_size,wConv,wRect, wNeuMaxAvg, wNeuBias,wHiddenMaxAvg,wHiddenBias,dropoutChoice,'maxavg','hidden')
    #loss=tf.reduce_mean(tf.losses.mean_squared_error(labelNetwork,pred))+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHiddenMaxAvg))+beta3*tf.reduce_sum(tf.abs(wNeuMaxAvg))
    #loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(pred,labelNetwork)))+beta1*tf.reduce_sum(tf.abs(wConv))+beta2*tf.reduce_sum(tf.abs(wHiddenMaxAvg))+beta3*tf.reduce_sum(tf.abs(wNeuMaxAvg))
    ##MAXAVGNOHIDDEN
    #pred = conv_net(inputNetworkBatch, batch_size,wConv,wRect, wNeuMaxAvg, wNeuBias,wHiddenMaxAvg,wHiddenBias,dropoutChoice,'maxavg','nohidden')
    #loss=tf.reduce_mean(tf.losses.mean_squared_error(labelNetwork,pred))+beta1*tf.reduce_sum(tf.abs(wConv))+beta3*tf.reduce_sum(tf.abs(wNeuMaxAvg))
        
    #########################################################################################################################################    
#    learning_rate=logsampler(0.005,0.05)
#    momentum_rate=sqrtsampler(0.95,0.99)    
    learning_rate=saveLearningRate
    momentum_rate=saveLearningMomentum

    #optimize with gradient descent with momentum, nesterov applied (paper say this)    
    #optimizer=tf.train.MomentumOptimizer(learning_rate,momentum_rate,use_nesterov=True).minimize(loss)
    ##optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
     #train_prediction = tf.nn.softmax(pred)
    
    accuracy=tf.reduce_mean(tf.abs(tf.subtract(pred,labelNetwork)))
#    valid_prediction1=conv_net(inputValid1,len(valid_dataset_tot[0]), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','validation')
#    valid_prediction2=conv_net(inputValid2,len(valid_dataset_tot[0]), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','validation')
#    valid_prediction3=conv_net(inputValid3,len(valid_dataset_tot[0]), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','validation')
#    valid_prediction=[valid_prediction1,valid_prediction2,valid_prediction3]

   # test_prediction=conv_net(tf_test_dataset, len(test_dataset), wConv,wRect, wNeuMaxHidden, wNeuBias,wHidden,wHiddenBias,dropoutChoice,'max','hidden','test')
    #testlab=tf.constant(test_labels,shape=[len(test_labels),1])
    #accurval=tf.reduce_mean(tf.abs(tf.subtract(tf.squeeze(test_prediction),labelTest)))

        #i did it, accuracy maybe it's the mean of this values
    
        #maybe I need these lines for SELEX and CHIP-seq
        #correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #def accuracy(predictions, labels):
        #  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    
    #def accuracy(predictions, labels):
    #    sum=0    
    #    for i in range(len(predictions)):
    #        if(predictions[i]==labels[i]):
    #            sum+=1
    #    return (100.0 * sum / len(predictions))

resa2=[]
out2=[]

with tf.Session(graph=graph2) as sess:
    sess.run(tf.global_variables_initializer()) #initialize variables
    print('Initialized all Variables')
    step=0
    lastphase=0
    termin=step *batch_size +batch_size
    start=0
    while termin  < len(alldataset):
        batch_data=[el[0] for el in alldataset[start:(start + batch_size)]] 
        batch_labels=[el[1] for el in alldataset[start:(start + batch_size)]]
        start=termin
        termin=termin+batch_size
        feeddict={inputNetworkBatch: batch_data, labelNetwork: batch_labels}
        sess.run(optimizer,feed_dict=feeddict)
        predictions,l,_,acc= sess.run([pred,loss,optimizer,accuracy],feed_dict=feeddict)
        resa.append(predictions)
        #out.append(wConv[0].eval())
        phase = int((termin)/len(alldataset)*100) 
        if (phase % 5 ==0) and (phase!=lastphase):
           #print("Minibatch loss at step %d %% : %.3f" % (phase, l))
           lastphase=phase
           print("Phase: %d %% \nAccuracy is: %f" % (phase,acc))
               #accval=np.average(np.abs(np.subtract(valid_prediction.eval(),valid_labels)))
    

            
    #open file
    
    #motiflen=checkSelex('Alx1_DBD_TAAAGC20NCG_3_Z_B.seq.gz')
    #iterations
   # np.random.shuffle(train_dataset)
#    step=0
#    lastphase=0
#    termin=step *batch_size +batch_size
#    start=0
#    
#    while termin  < len(train_dataset):
#        batch_data=[el[0] for el in train_dataset[start:(start + batch_size)]] 
#        batch_labels=[el[1] for el in train_dataset[start:(start + batch_size)]]
#        start=termin
#        termin=termin+batch_size
#        feeddict={inputNetworkBatch: batch_data, labelNetwork: batch_labels}
#        sess.run(optimizer,feed_dict=feeddict)
#        predictions,l,_,acc= sess.run([pred,loss,optimizer,accuracy],feed_dict=feeddict)
#        resa.append(predictions)
#        #out.append(wConv[0].eval())
#        phase = int((termin)/len(train_dataset)*100) 
#        if (phase % 5 ==0) and (phase!=lastphase):
#           #print("Minibatch loss at step %d %% : %.3f" % (phase, l))
#           lastphase=phase
#           print("Phase: %d %% \nAccuracy is: %f" % (phase,acc))
#           accval=np.average(np.abs(np.subtract(valid_prediction.eval(),valid_labels)))
#           acctest=np.average(np.abs(np.subtract(test_prediction.eval(),test_labels)))
#           print('Accuracy of test and validation is: %f , %f ' % (accval,acctest))
        #out = wConv.eval(session=sess)



#           print('fammi veder ',wConv.eval(session=sess))
#               print('pred ',predictions)               
#               print('lab ', batch_labels)
#               print(np.average(abs(np.subtract(batch_labels,predictions))))
        #step=step+1
        
        
#    print('wConv ', wConv.eval())

    
#    print('wRect ', wRect.eval())
#    print('wNeu ', wNeu.eval())
#    print('wNeuBias ', wNeuBias.eval())


