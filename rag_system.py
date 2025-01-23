import numpy as np
from sklearn.manifold import TSNE  # Libreria per ridurre la dimensionalità dei dati
import matplotlib.pyplot as plt  # Libreria per creare grafici e visualizzazioni
from sklearn.metrics.pairwise import cosine_similarity  # Funzione per calcolare la similarità coseno tra vettori
from FlagEmbedding import BGEM3FlagModel  # Modello per calcolare embeddings vettoriali


# Caricamento del modello BGEM3FlagModel
# Utilizza il modello 'BAAI/bge-m3' con precisione a 16 bit per migliorare le prestazioni
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def index_database(data):
    """
    Calcola gli embeddings per i dati forniti e li salva in un file .npy.

    Args:
        data (list): Lista di frasi o dati testuali per cui calcolare gli embeddings.
    """
    # Calcola gli embeddings delle frasi
    embeddings = model.encode(data)['dense_vecs']
    # Salva gli embeddings in formato numpy per un accesso rapido
    np.save('embeddings.npy', embeddings)

def load_embedding_matrix(dembeddings_path):
    """
    Carica una matrice di embeddings salvata precedentemente in un file .npy.

    Args:
        dembeddings_path (str): Percorso del file .npy contenente gli embeddings.

    Returns:
        numpy.ndarray: Matrice di embeddings caricata.
    """
    # Carica gli embeddings dal file .npy
    loaded_embeddings = np.load(dembeddings_path)
    return loaded_embeddings

def search(query, embedding_matrix):
    """
    Cerca una query nel database vettoriale calcolando la similarità coseno.

    Args:
        query (str): Testo della query da cercare.
        embedding_matrix (numpy.ndarray): Matrice di embeddings pre-calcolati.

    Returns:
        list: Lista ordinata di risultati con indici e punteggi di similarità.
    """
    # Calcola l'embedding vettoriale della query
    query_embedding = model.encode([query])['dense_vecs'][0]
    # Calcola la similarità coseno tra la query e tutti gli embeddings nella matrice
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
    # Ordina i risultati per similarità decrescente e restituisce gli indici
    similarity_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    return similarity_results

def visualize_space_query(data, query, embedding_matrix):
    """
    Visualizza gli embeddings delle frasi e la query in uno spazio bidimensionale
    utilizzando t-SNE.

    Args:
        data (list): Lista di frasi originali.
        query (str): Testo della query.
        embedding_matrix (numpy.ndarray): Matrice di embeddings delle frasi.
    """
    # Calcola l'embedding della query
    query_embedding = model.encode([query])['dense_vecs'][0]
    # Combina gli embeddings delle frasi con quello della query
    jointed_matrix = np.vstack([embedding_matrix, query_embedding])

    # Riduzione dimensionale a due dimensioni con t-SNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(jointed_matrix)

    # Configurazione del grafico
    plt.figure(figsize=(8, 6))

    # Plot degli embeddings delle frasi come punti blu
    plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], c='blue', edgecolor='k', label='Frasi')

    # Plot dell'embedding della query come punto rosso
    plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], c='red', edgecolor='k', label='Query')

    # Annotazioni delle frasi
    for i, frase in enumerate(data):
        plt.text(embeddings_2d[i, 0] + 0.1, embeddings_2d[i, 1] + 0.1, frase, fontsize=9)

    # Annotazione per la query
    plt.text(embeddings_2d[-1, 0] + 0.1, embeddings_2d[-1, 1] + 0.1, query, fontsize=9, color='red')

    # Dettagli del grafico
    plt.title('Visualizzazione degli Embeddings con t-SNE')
    plt.xlabel('Dimensione 1')
    plt.ylabel('Dimensione 2')
    plt.grid(True)
    plt.legend()
    plt.show()

# Frasi di esempio (commentato per utilizzo opzionale)
# frasi = [
#     "Come migliorare il proprio benessere mentale con la meditazione.",
#     "I benefici di una dieta equilibrata e ricca di proteine.",
#     "L'importanza di fare esercizio fisico regolare per la salute cardiovascolare.",
#     "Nuovi progressi nella ricerca contro il cancro con l'intelligenza artificiale.",
#     "Come prevenire problemi alla schiena con una postura corretta.",
#     "La scoperta di un farmaco innovativo per la gestione del diabete di tipo 2.",
#     "Rimedi naturali per migliorare la qualità del sonno."
# ]

# Esempio di query per la ricerca
# query = "Come posso migliorare la mia salute cardiovascolare?"

# Esecuzione delle funzioni principali (commentate come esempio)
# index_database(frasi)  # Calcola e salva gli embeddings
# matrix = load_embedding_matrix("embeddings.npy")  # Carica gli embeddings salvati
# out = search(query=query, embedding_matrix=matrix)  # Ricerca della query
# print(out)  # Stampa i risultati
# visualize_space_query(frasi, query, matrix)  # Visualizza i risultati
