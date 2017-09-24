# Problemi 
1. La lunghezza delle sequenze di RNACompete è variabile. Questo è un problema perché il codice viene eseguito tramite batch di dati che devono avere la stessa dimensione di input, anche perché nei tensori non è possibile una dinamicità nella forma. Per questo ho ovviato trovando la lunghezza della sequenza più grande, effettuando padding (inserendo N) nelle sequenze più corte fino ad arrivare alla stessa lunghezza
2. Per ogni variabile da fittare c'è uno spazio di ricerca, mentre per le soglie di rettificazione no. Come impostarla?
3. Ho usato al posto del max pooling e dell'average pooling le operazioni di reduce max e mean perché supportano una dimensione diversa delle sequenze. Non avendone bisogno, si può tornare ll'utilizzo di maxpool e avgpool.
4. Nel paper utilizzano l'algoritmo SGD, settando il momento e l'utilizzo di Nesterov. SGD non ha questi due input, mentre MomentumOptimizer e RMSPropOptimizer sì. Quale usare?
5. Problema di fondo: io credo e sono abbastanza convinto che il modello si debba allenare per ogni Fattore di Trascrizione, perchè le previsioni devono essere differenti per ognuno di essi.
6. 
