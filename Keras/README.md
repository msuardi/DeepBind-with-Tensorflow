# SVILUPPO IN KERAS
Le varie parti del codice vengono spiegate all'interno del notebook del codice stesso; prima di passare alla sua lettura, è necessario capire quali sono i dati in input, i problemi riscontrati, e i futuri sviluppi.

## Dati in input
DeepBind è un software in grado di effettuare predizioni sulla specificità dei fattori di trascrizione su sequenze genomiche (sia DNA che RNA). Per farlo, allena un modello prendendo dei dati in input di diversa forma.
Infatti, la specificità può essere dedotta a partire da differenti esperimenti, e gli autori del paper originali hanno voluto creare un software unico per l'elaborazione di dati provenienti da ciascuna fonte.
Scaricando i dati supplementari, troviamo una cartella contenente il codice e una contenente i dati.
Questi dati però sono già stati elaborati, ovvero sono stati rimossi quelli poco significativi / outliers, sono stati suddivisi in training e test, e sono già presenti le predizioni da loro ottenute.
Per questo, se possibile (come nel caso di PBM), ho scaricato i dati originali, mentre negli altri tre casi non sono rintracciabili i dati originali.

### PBM
PBM (Protein Binding Microarrays) è un esperimento nel quale si ottiene, per ogni fattore di trascrizione, la specificità (un float in un intervallo indefinito) di ciascuna sequenza di DNA ricercata tramite l'utilizzo di microarray fisici.
Gli autori del paper hanno utilizzato i dati provenienti da una competizione, DREAM5 TF-DNA Motif Recognition Challenge (del 2003) per allenare il modello ed effettuare le previsioni.
È possibile, dal sito della competizione (http://dreamchallenges.org/project/dream-5-tf-dna-motif-recognition-challenge/), scaricare i dati originali di training e test che sono così realizzati.
Nel file di training, sono presenti le specificità di 20 fattori di trascrizione.
In particolar modo, per ogni fattore di trascrizione di training sono presenti le specificità di circa 80000 sequenze, 40000 delle quali allenate con un tipo di array (HK) e 40000 con un altro (ME).
Per ogni fattore di trascrizione queste 80000 sequenze sono sempre le stesse, con ovviamente diversi valori di specificità.
La "sfida" richiedeva di predire le specificità di altri 66 fattori di trascrizione per i quali è presente la specificità di un tipo di array (HK o ME) e non dell'altro tipo.
Per questo, essendo disponibili al download questi dati, ho pensato di utilizzare i dati originali stessi.
Dovendo predire le specificità di un fattore di trascrizione X per l'esperimento ME (noti quindi i risultati per HK), ho deciso di allenare un modello per ciascuno dei 20 fattori di trascrizione del training, usando come validation set l'insieme delle specificità di X per l'esperimento HK.
Avendo il risultato di validation error migliore, ho predetto la specificità rimanenti usando quel modello stesso.
In questo caso ho modificato le specificità normalizzando ciascun punteggio, in quanto non facendolo portava ad avere tutte le predizioni uguali.
I dati forniti dagli autori sono differenti, suddivisi in due fold e con valori di lunghezza delle sequenze e specificità diversi, ho preferito perciò usare i dati originali.


### RNACompete
RNACompete è un esperimento simile al precedente, utilizzando però sequenze di RNA. 
I dati non sono disponibili online, perciò ho utilizzato quelli scaricati dai file supplementari del paper.
Sono presenti due file: sequences.tsv e targets.tsv.
Nel file sequences sono presenti le sequenze di RNA, divisi in due fold A e B, che corrispondono al training e al test set; in targets ci sono invece tutte le specificità, divise per fattore di trascrizione.
I dati di validation sono ottenuti da una determinata frazione passata in input del training set.
Da notare che le sequenze di input sono di differente lunghezza: per ragioni implementative ho effettuato il padding di ciascuna sequenza alla lunghezza massima di una sequenza utilizzando la non-base 'N'.

## CHIP-seq e SELEX
Neanche in questo caso i dati scaricabili si presentano in una forma commprensibile, per questo ho utilizzato i dati già presenti.
Questi sono suddivisi in due file: le sequenze di input e relative specificità, sequenze di output e relative specificità.
In questo caso le specificità sono invece semplici valori binari: 0 sta per non specifico, 1 per specifico.
In particolar modo tra i dati di training le prime 500 sequenze sono quelle più specifiche tra tutte queste (informazione presente nel paper, allenare il modello solo con le prime 500 o con tutte è il mio dubbio), mentre non sono presenti sequenze di training a specificità 0.
Per questo tramite dinucleotide-shuffle (ovvero shuffle di ciascuna sequenza di input mantenendo intatte le coppie di basi), si generano le sequenze a specificità zero.
Le stesse identiche operazioni vanno ripetute per lo studio di dati provenienti da esperimenti di tipo SELEX, con i dati che sono presenti nelal stessa forma di CHIP-seq (cambia solo la lunghezza delle sequenze di ciascun TF, variabile, alla quale è strettamente collegata la lunghezza del motif detector, concetto che viene spiegato nel codice)

## DEEPFIND
Come ultima parte del lavoro, gli autori del paper hanno realizzato un programma chiamato DeepFind, il cui compito è quello di capire quanto può essere deleteria una variazione di un nucleotide in una sequenza genomica.
Per farlo hanno considerato delle SNV (Single-Nucleotide Variation) presenti in una ricerca precedente, e hanno suddiviso i dati in due cartelle: quelle dei dati derivati (cioè in cui sono presenti SNV corrispondenti a variazioni di alleli, non dannose) e quelle dei dati simulati (corrispondenti alle potenzialmente pericolose SNV).
A questo punto hanno calcolato il punteggio di queste sequenze utilizzando CHIP e Selex, per poi assegnare ai primi dati etichetta 0, ai secondi 1.
Usando come input le specificità e come target 0/1, hanno creato una rete neurale per determinare quanto bene fossero realizzati i modelli CHIP e Selex, ottenendo un'alta AUC, 0.71.
In questo campo ho modificato il file da loro scritto in modo da essere compatibile con tensorflow, ottenendo risultati simili mantenendo un basso numero di epoche; alzandole infatti la loss tende a nan.

## PROSSIMI STEP
- Valutare la bontà degli esperimenti
- Testare in parallelo seguendo ciò che ha fatto Cardillo / usando multi_gpu /usando aws
- Installare CUDA9 e testare
- Sentire Costello per i dati PBM
- Sentire autori per PBM e RNAc evaluation 

## NOTE AGGIUNTIVE
Per evitare di proseguire nelle epoche con loss nan, ho inserito una callback, che fa early-termination in caso la loss finisca a nan. 
Inoltre ho inserito un'altra callback, chiamata tensorboard:
TensorBoard(log_dir='./', histogram_freq=0,write_graph=True, write_images=True)
Così facendo posso poi eseguire nella cartella corrente nella shell "tensorboard --logdir=./", accedere a localhost:6006 e trovare una rappresentazione grafica interessante del training effettuato.

