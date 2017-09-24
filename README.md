# DeepBind-with-Tensorflow
L'obiettivo della tesi è di riproporre il lavoro di Alipanahi, Delong, Weirauch e Frey, spiegato nel paper "Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning" con l'utilizzo di Tensorflow.
Il paper è disponibile con accesso full text al link seguente:
http://www.nature.com/nbt/journal/v33/n8/full/nbt.3300.html

# Operazioni preliminari
Per poter eseguire i codici presenti nel repository, è necessario installare Tensorflow e tutti i requisiti necessari al suo funzionamento. Seguono alcune indicazioni su ciò che ho fatto e alcune guide seguite per velocizzare il tutto.

## Installazione CUDA
Il mio PC è dotato di una scheda video NVIDIA GEForce 930MX, e sto lavorando su Linux Ubuntu 16.04 LTS.
Ho seguito la guida presente a questo link: 
http://docs.nvidia.com/cuhttp://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installationda/cuda-installation-guide-linux/index.html#ubuntu-installation

Ovvero:
- Ho scaricato i pacchetti per Linux Ubuntu 16.04 x86_64
- sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
- sudo apt-get update
- sudo apt-get install cuda
- export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
- export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} in user and root
- source /etc/environment sia nell'utente che in root
- cd /usr/local/cuda-8.0/samples e quindi ho eseguito il makefile
- Ho provato a caricare un sample, come nbody (presente in 5-Simulations)
- Dovesse funzionare, allora l'installazione è andata a buon fine
- Se si verifica un errore come: no device in /dev/nvidia, vuol dire che è necessario un aggiornamento del driver Nvidia
In questo caso ho eseguito le operazioni seguenti:
- sudo add-apt-repository ppa:graphics-drivers/ppa
- sudo apt-get update
- sudo apt-get install nvidia-381
- error: cuda driver version is insufficient for cuda runtime version
- sudo apt-get install nvidia-378
- Ora è andato tutto a buon fine

# Installazione CUDNN v5.1
- download libcudnn5_5.1.10-1+cuda8.0_ppc64el.deb
- download libcudnn5.1 libraries
- tar -xvzf cudnn-8.0-linux-x64-v5.1.tgz cuda/
- cd folder/extracted/contents
- sudo cp -P include/cudnn.h /usr/include
- sudo cp -P lib64/libcudnn- /usr/lib/x86_64-linux-gnu/
- sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn-

# Installazione di Tensorflow
A questo punto è possibile installare Tensorflow, seguendo la guida a questo link: https://www.tensorflow.org/install/install_linux

## Download dati di DeepBind
A fondo pagina, o cliccando questo link si possono scaricare il codice e i dati utilizzati nel paper originale: http://www.nature.com/nbt/journal/v33/n8/extref/nbt.3300-S13.zip

# Esecuzione del codice originale --> NON RIUSCITA
Per poter avere un confronto più immediato ho provato a eseguire il codice scaricato ma ci sono dei problemi non risolvibili neanche contattando gli autori del paper stesso.
Infatti nella cartella principale c'è un README con le operazioni da eseguire, che consentono di compilare tutte le librerie create appositamente, tra cui:
- cd libs/smat/src
- make
- cd ../py
- python run_tests.py
- cd ../../deepity/deepity_smat
- make 
- cd ../../kangaroo/kangaroo_smat
- make

L'operazione _python run_tests.py_ però non va a buon fine, si verifica un assertion error di questo tipo: _smat_zeros function doesn't create an array full of zero_ .
Contattando gli autori, Andrew Delong mi ha suggerito di cambiare il makefile della libreria smat, poiché era configurato per una vecchia versione di CUDA; cambiando CUDA_ARCH settandolo a valore 50.
A questo punto tutti i test vengono passati.
L'operazione di make in deepity non crea problemi, mentre quella in kangaroo_smat solleva il seguente errore:

/usr/local/cuda/include/cuda.h(60): error: "uint64_t" is ambiguous
1 error detected in the compilation of "/tmp/tmpxft_000017b6_00000000-7_autotune.cpp1.ii".
makefile:159: recipe for target '../build/release/obj/KANGAROO_SMAT/autotune.o' failed
make: [../build/release/obj/KANGAROO_SMAT/autotune.o] Error 2

A questo punto sia Delong che Alipanahi hanno ritenuto che questo fosse un errore dovuto al fatto che il codice era stato realizzato con una versione vecchia di CUDA, e quindi potrebbe essere non compatibile con la versione attuale a causa di una qualche collisione.

# Codice
A questo punto sono passato alla scrittura del codice, seguendo il paper originale  e quello contenente le informazioni supplementari: https://images.nature.com/full/nature-assets/nbt/journal/v33/n8/extref/nbt.3300-S2.pdf

## Codice in Tensorflow
Dapprima ho provato a realizzare il tutto in Tensorflow, eseguendo ciò che avevano fatto gli autori passo dopo passo.
Nonostante ciò incappavo sempre in un problema: tutte le predizioni risultavano uguali.
Ciò che ho fatto e le istruzioni per eseguirlo sono presenti nella cartella Tensorflow.

## Codice in Keras
Cercando una soluzione, ho trovato Keras, una API scritta in Python in grado di eseguirsi sopra Tensorflow, la quale rende tutte le operazioni delle reti neurali più rapide e intutitive.
Ho perciò riscritto il codice utilizzando Keras.
Ciò che ho fatto e le istruzioni per eseguirlo sono presenti nella cartella Keras.


