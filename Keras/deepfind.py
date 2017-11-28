from os import listdir
from os.path import join
import numpy as np
from keras import optimizers
from keras.layers import Dense,Dropout
from keras.models import Sequential
from util import log_loss,logsampler,sqrtsampler,calc_auc
import math
from multi_gpu import to_multi_gpu


#cartella di origine dei file necessari
data_root='../data/deepfind'
batchsize=128
dt=np.float32
num_epochs=250 #TODO set to 250 later
learning_rate=logsampler(0.005,0.05)
momentum_rate=sqrtsampler(0.95,0.99)

# input files and directories
pos_dir        = join(data_root, 'tfs_simulated')
neg_dir        = join(data_root, 'tfs_derived')
cons_pos_file  = join(data_root, 'simulated_cons.npz')
cons_neg_file  = join(data_root, 'derived_cons.npz')
mask_file      = join(data_root, 'matching_mask.npz')
pos_feat_file  = join(data_root, 'simulated_feats.npz') 
neg_feat_file  = join(data_root, 'derived_feats.npz')

#operazioni opzionali
add_cons   = True  # add conservation information
add_feats  = True  # add dist to TSS and is_transversion features
add_tfs    = True  # add predictions from TFs for wildtype and mutant
do_masking = True  # match genes of positive and negative sets


#il file TFs_to_consider contiene l'insieme dei TF da considerare (per le prove utilizzato set minore, memory error)
tfs_to_consider = join(data_root, 'TFs_to_consider.txt')

with open(tfs_to_consider) as fh:
    considered_files = [line.strip('\r\n') for line in fh]

#funzione per caricare i dati
def load_data(save_dir, considered_files=None):
    files = listdir(save_dir)
    factor_names = []
    
    if considered_files is None: #se il file TFs_to_consider non esiste,utilizzo tutti i file della cartella
        dim = 2*(len(files)) #per ogni TF si considera la wild type prediction e la mutant-wildtype
    else:
        dim = 2*(len(considered_files))

    cnt = 0
    for file_name in sorted(files):
        if considered_files is not None and file_name[:-4] not in considered_files:
            continue #si ignorano i TF da non considerare
            
        factor_names.append(file_name[:-4]) # si aggiunge ai factor_names considerati quello corrente (si toglie l'estensione .npz)
        print('id is  ', join(save_dir,file_name))
        with np.load(join(save_dir, file_name)) as data:
            p = data['pred']
            if cnt == 0:
                # si inizializza la matrice delle feature
                X = np.empty((int(p.shape[0]/2),dim)) 
            X[:,2*cnt]   = p[::2]               # wild type predictions
            X[:,2*cnt+1] = p[1::2] - p[::2]     # mutant - wildtype predictions
        cnt += 1    
    return X, factor_names

pX, pfactors = load_data(pos_dir, considered_files) #questi sono i dati corrispondenti ai TF simulati, cioè le vere e proprie varianti dannose
nX, nfactors = load_data(neg_dir, considered_files) #questi sono i dati derivati, ovvero le normali alterazioni degli alleli

print('Adding prediction for %d TFs' % len(pfactors))
for pf, nf in zip(pfactors, nfactors):
    if not pf == nf:
        print('Mismatched TFs!')

#si combinano le predizioni positive e negative
X = np.vstack([pX, nX])
#si associano alle prime l'etichetta 1, all seconde l'etichetta 0
Y = np.vstack([np.ones((pX.shape[0],1)), np.zeros((nX.shape[0],1))])

# Si aggiungono le informazioni di "conservazione", se richiesto
if add_cons:
    print('Adding conservation')
    with np.load(cons_pos_file) as data:
        pC = data['cons']
    with np.load(cons_neg_file) as data:
        nC = data['cons']                 
    C = np.vstack((pC, nC)) #si combinano i due array
    # add conservation information to TF features
    X = np.hstack((X, C)) #si aggiungono le informazioni di conservazione


# Si aggiungono il transversion_flag per le mutazioni e la distanza normalizzata dal più vicino TSS (da 0 a 1) se richiesto
if add_feats:
    print('Adding two extra features')
    with np.load(pos_feat_file) as data:
        pF = data['feats']
    with np.load(neg_feat_file) as data:
        nF = data['feats']        
    X = np.hstack([X, np.vstack([pF, nF])]) #come prima nella conservazione 
        
# si applica il masking, togliendo alcuni dei TF, se richiesto
if do_masking:
    print('Matching genes')
    with np.load(mask_file) as data:
        c_mask = np.hstack([data['pos_mask'], data['neg_mask']])                            
    X = X[c_mask]
    Y = Y[c_mask]
    
num, dim = X.shape #si salvano in num e dim le dimensioni di X              
print('Data is loaded\nsample size: %d, dimensions:%d' % (num, dim))

#si effettua una permutazione casuale dei dati 
np.random.seed(1234)
shuffled_idx = np.random.permutation(num)
X[:] = X[shuffled_idx]
Y[:] = Y[shuffled_idx]
   
#nel paper viene richiesta una five-fold cross validation


#split dei dati in base al fold_id    
def split_data(fold_id, num_fold):
    sp = np.linspace(0, num, num_fold+1).astype(np.int)
    splits = np.empty((num_fold, 2), dtype=np.int)
    for i in range(num_fold):
        splits[i,:] = [sp[i], sp[i+1]]
    all_splits = set(np.arange(num_fold))
    ts = np.mod(num_fold-1+fold_id, num_fold)
    vd = np.mod(num_fold-2+fold_id, num_fold)
    tr = list(all_splits - set([ts, vd]))
    idx_ts = np.arange(splits[ts,0],splits[ts,1])
    idx_vd = np.arange(splits[vd,0],splits[vd,1])
    idx_tr = np.arange(splits[tr[0],0],splits[tr[0],1])
    for i in range(1, len(tr)):
        idx_tr = np.hstack([idx_tr, np.arange(splits[tr[i],0],splits[tr[i],1])])
    
    Xtr = np.asarray(X[idx_tr])
    Ytr = np.asarray(Y[idx_tr])
    Xts = np.asarray(X[idx_ts])
    Yts = np.asarray(Y[idx_ts])
    Xvd = np.asarray(X[idx_vd])
    Yvd = np.asarray(Y[idx_vd])

    return Xtr,Ytr,Xts,Yts,Xvd,Yvd
    
    
#############################################################
    
def findNet(inpShape):
    model = Sequential()
    model.add(Dense(200,input_shape=(inpShape,),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    return model
    

def deepFind():
    num_fold=5
    list_auc=list()
    test_auc=list()
    for fold_id in range(num_fold):
        test_fold = np.mod(num_fold-1+fold_id, num_fold)+1
        print("*******************************************************************")
        print("Training a neural network and testing on fold %d" % (test_fold))
        Xtr,Ytr,Xts,Yts,Xvd,Yvd=split_data(fold_id,num_fold)
        model = findNet(dim)
        model = to_multi_gpu(model,40)
        model.compile(loss=log_loss,
                          optimizer=optimizers.SGD(lr=learning_rate,momentum=momentum_rate,nesterov=True,decay=1e-6))
        print('Ready to Fit OK')
        best_auc=0
        best_model=None
        list_auc_in=list()
        for i in range(num_epochs):
            print('\nFitting in epoch %d' %(i))
            model.fit(Xtr, Ytr,
                          batch_size=batchsize*40,epochs=1,
                          verbose=1)
            prediction = model.predict(Xvd,batchsize,verbose=1)
            print('\n')
            prediction = np.reshape(prediction,[prediction.shape[0]])
            auc = calc_auc(prediction,Yts)
            list_auc_in.append(auc)
            if math.isnan(auc):
                print('\n This training went nan')
            if auc>best_auc:
                best_auc=auc
                best_model=model
        print('\nFit complete. Now score and prediction')
        prediction = best_model.predict(Xts,batchsize,verbose=1)
        prediction = np.reshape(prediction,[prediction.shape[0]])
        auc = calc_auc(prediction,Yts)
        test_auc.append(auc)
        list_auc.append(list_auc_in)
    return list_auc,test_auc
    
listauc,testauc=deepFind()
print('\n Average test auc is %2f \n' %(np.mean(testauc)))