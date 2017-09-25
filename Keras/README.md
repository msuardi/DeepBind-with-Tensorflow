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

## 
