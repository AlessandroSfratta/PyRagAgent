import streamlit as st  # Libreria per creare interfacce web interattive e leggere
import openai  # Libreria per interagire con l'API OpenAI
import time  # Per gestire ritardi temporali durante le operazioni
import json  # Per manipolare e leggere dati in formato JSON
from typing import Dict, List  # Per annotazioni di tipo più leggibili
from dotenv import load_dotenv  # Per caricare variabili di configurazione da file .env
import os  # Per accedere e gestire variabili di ambiente e operazioni di sistema

from email_utils import cerca_mail  # Funzione personalizzata per effettuare ricerche di email

# Caricamento delle variabili di configurazione definite nel file .env
load_dotenv()
openai_api_key = os.environ['OpenAI_API_Key']  # Legge la chiave API di OpenAI dalla variabile di ambiente

# Configura l'aspetto e il comportamento dell'app Streamlit
st.set_page_config(page_title="Assistant API Chatbot", page_icon="🤖")

# Inizializzazione delle variabili di sessione per mantenere lo stato tra i ricaricamenti dell'app
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Lista per salvare i messaggi della conversazione corrente

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None  # Identificatore univoco per il thread di conversazione

# Funzione per cercare email in base a una query
def search_emails(query: str) -> List[Dict]:
    """
    Esegue una ricerca di email che corrispondono alla query fornita.
    In questa implementazione, utilizza la funzione `cerca_mail`.

    Args:
        query (str): La query di ricerca dell'utente.

    Returns:
        List[Dict]: Lista di risultati delle email trovate.
    """
    print(f"Cerca la mail con la query: {query}")
    return cerca_mail(query)  # Simula la ricerca delle email utilizzando una funzione esterna

# Configurazione delle chiavi API e dell'ID dell'assistente OpenAI
OPENAI_API_KEY = openai_api_key  # Chiave API necessaria per autenticare le richieste a OpenAI
ASSISTANT_ID = "asst_ospySOMp9lqoJ42gq1wtosqQ"  # ID dell'assistente specifico configurato su OpenAI

# Inizializzazione del client OpenAI utilizzando la chiave API
client = openai.Client(api_key=OPENAI_API_KEY)

def create_thread():
    """
    Crea un nuovo thread di conversazione per mantenere il contesto tra i messaggi inviati.

    Returns:
        str: Identificatore univoco del thread creato.
    """
    thread = client.beta.threads.create()  # Crea un nuovo thread tramite l'API OpenAI
    return thread.id  # Ritorna l'ID del thread

def process_message(user_input: str, thread_id: str):
    """
    Elabora un messaggio utente, invia la richiesta al modello e ottiene la risposta generata.

    Args:
        user_input (str): Messaggio dell'utente da elaborare.
        thread_id (str): ID del thread per mantenere il contesto della conversazione.

    Returns:
        str: Risposta generata dall'assistente.
    """
    # Aggiunge il messaggio dell'utente al thread esistente
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )
    
    # Richiede l'elaborazione dell'assistente
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )
    
    # Loop per monitorare lo stato del run fino al completamento
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
        if run_status.status == 'requires_action':  # Se il modello richiede una funzione specifica
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            
            for tool_call in tool_calls:
                if tool_call.function.name == "search_emails":
                    # Estrazione degli argomenti della funzione
                    arguments = json.loads(tool_call.function.arguments)
                    query = arguments.get("query", "")  # Recupera la query dalla funzione
                    
                    # Esegue la funzione personalizzata di ricerca email
                    result = search_emails(query)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,  # ID della chiamata funzione
                        "output": str(result)  # Risultato della funzione
                    })
            
            # Invia i risultati delle funzioni al modello
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        
        elif run_status.status == 'completed':  # Se l'elaborazione è completata
            break  # Esce dal ciclo di monitoraggio
        
        elif run_status.status in ['failed', 'cancelled']:  # Gestione errori o cancellazione
            st.error(f"Run failed or cancelled: {run_status.status}")
            return None
        
        time.sleep(1)  # Introduce un breve ritardo per evitare chiamate eccessive
    
    # Recupera l'ultimo messaggio generato dall'assistente
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return messages.data[0].content[0].text.value  # Restituisce il testo della risposta

def main():
    """
    Punto di ingresso principale per l'app Streamlit.
    Gestisce l'interfaccia utente e la logica della conversazione.
    """
    st.title("📧 Email Assistant Chatbot")  # Titolo dell'app
    st.write("Chiedimi qualsiasi cosa riguardo le tue email!")  # Sottotitolo/istruzioni

    # Crea un nuovo thread di conversazione se non esiste già
    if not st.session_state.thread_id:
        st.session_state.thread_id = create_thread()

    # Input dell'utente tramite il componente chat di Streamlit
    user_input = st.chat_input("Scrivi il tuo messaggio qui...")

    # Visualizza la cronologia dei messaggi nella conversazione
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # Mostra il ruolo (utente o assistente)
            st.write(message["content"])  # Mostra il contenuto del messaggio

    # Processa un nuovo messaggio utente se fornito
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)  # Mostra il messaggio dell'utente nell'interfaccia
        st.session_state.messages.append({"role": "user", "content": user_input})  # Aggiunge il messaggio alla cronologia

        # Genera la risposta dell'assistente
        with st.chat_message("assistant"):
            response_placeholder = st.empty()  # Spazio per la risposta in caricamento
            response = process_message(user_input, st.session_state.thread_id)  # Ottiene la risposta
            response_placeholder.write(response)  # Mostra la risposta dell'assistente
        st.session_state.messages.append({"role": "assistant", "content": response})  # Salva la risposta nella cronologia

if __name__ == "__main__":
    main()  # Avvia l'applicazione
