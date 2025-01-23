from rag_system import * 
import os
import json

# Lista per memorizzare i contenuti delle email
frasi = []

# Scansione dei file nella directory "data"
for f in os.listdir("data"):
    # Apertura di ciascun file JSON nella directory
    with open(f"data/{f}", "r", encoding="utf-8") as jsonf:
        # Lettura del contenuto del file JSON e aggiunta del campo 'body' alla lista 'frasi'
        frasi.append(json.loads(jsonf.read())['body'])





# Caricamento della matrice di embedding dai dati salvati in un file .npy
matrix = load_embedding_matrix("embeddings.npy")
# Ricerca della query "Collaborazione inferentia" nel database vettoriale
out = search(frasi, "Collaborazione inferentia", matrix)
# Stampa il quarto risultato della ricerca (indice 3)
print(out[3])
