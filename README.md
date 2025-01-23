# PyRagAgent

## Cosa fa il progetto

PyRagAgent è un sistema semplice progettato per gestire email, documenti e database vettoriali utilizzando l'intelligenza artificiale (AI).\
&#x20;Questo progetto integra diverse tecnologie come embedding, indicizzazione, modelli linguistici (LLM), visualizzazioni di dati e supporto per le function calling di OpenAI, offrendo una piattaforma altamente funzionale   e progettata per organizzare e interagire con grandi volumi di dati in modo intelligente.



### Caratteristiche principali

1. **Database vettoriali**: Creazione e gestione di database ottimizzati con embeddings vettoriali per ricerche semantiche avanzate.
2. **Embedding personalizzati**: Utilizzo del modello `BGEM3FlagModel` per calcolare rappresentazioni vettoriali dei dati.
3. **Agente AI**: Un chatbot interattivo integrato con OpenAI per rispondere a domande, analizzare dati e gestire email.
4. **Indicizzazione dei dati**: Processi automatizzati per la creazione di database vettoriali e la ricerca semantica.
5. **Function Calling**: Configurazione e utilizzo di funzioni chiamabili all'interno di OpenAI per l'interazione dinamica con i dati.
6. **Visualizzazione**: Utilizzo di t-SNE ([Approfondisci qui](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)) per rappresentare graficamente le relazioni semantiche tra query e dataset.

---

## Installazione

### Requisiti

- Python >= 3.8
- Git
- Ambiente virtuale (consigliato per evitare conflitti di dipendenze)

### Passaggi di installazione

1. **Clona il repository**:

   ```bash
   git clone https://github.com/tuo-repo/PyRagAgent.git
   cd PyRagAgent
   ```

2. **Crea e attiva un ambiente virtuale**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Su Windows: .\venv\Scripts\activate
   ```

3. **Installa le dipendenze**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Installa il modello \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*****`FlagEmbedding`**:

   ```bash
   git clone https://github.com/FlagOpen/FlagEmbedding.git
   cd FlagEmbedding
   pip install -e .
   cd ..
   ```

5. **Configura le variabili di ambiente**:
   Crea un file `.env` nella directory principale con la tua chiave API di OpenAI:

   ```env
   OpenAI_API_Key=YOUR_API_KEY
   ```



---

## Configurazione

1. **OpenAI API**:

   - Registra un account su OpenAI e ottieni una chiave API seguendo le istruzioni ufficiali: [OpenAI API Key](https://platform.openai.com/signup/).
   - Inserisci la chiave API nel file `.env`.

2. **Function Calling di OpenAI**:

   - Assicurati di configurare correttamente l'`Assistant ID`.\
     &#x20;Puoi testare e configurare il tuo Assistant ID direttamente nel [Playground di OpenAI](https://platform.openai.com/playground/chat?models=gpt-4o-mini\&models-gpt-40-mini).&#x20;

3. **Dataset di email**:

   - Il dataset di email contiene file JSON strutturati con campi come `date`, `subject`, `from` e `body`. Ogni email rappresenta un messaggio completo, utile per l'addestramento e la ricerca semantica.
   - Puoi generare un nuovo dataset usando lo script `generate_emails.py`. Tuttavia, nella repository è già presente un dataset nella cartella `data`, pronto per l'uso. Se necessario, puoi rigenerarlo con il comando:
     ```bash
     python3 generate_emails.py
     ```

---

## Come usarlo

1. **Carica i tuoi dati**:
   Inserisci le email nella directory `data` in formato JSON. Ogni file deve contenere un campo `body` che rappresenta il contenuto dell'email.

2. **Indicizza i dati**:
   Usa il file `rag_system.py` per calcolare gli embeddings delle email e creare un database vettoriale:

   ```bash
   python3 index_data.py
   ```

3. **Esegui l'app Streamlit**:
   Avvia il server Streamlit per l'interfaccia utente:

   ```bash
   streamlit run agent_ai.py
   ```

4. **Interagisci con il chatbot**:
   Scrivi domande nell'interfaccia Streamlit per ottenere risposte basate sui dati indicizzati.

5. **Visualizza i risultati**:
   Usa la funzione di visualizzazione per analizzare graficamente le relazioni tra query e dati:

   ```python
   visualize_space_query(data, query, embedding_matrix)
   ```



---

## Tecnologie utilizzate

- **Numpy**: Per calcoli numerici e gestione delle matrici.
- **Scikit-learn**: Per il calcolo della similarità coseno e la riduzione dimensionale (t-SNE).
- **Matplotlib**: Per la visualizzazione grafica dei risultati.
- **Streamlit**: Per creare un'interfaccia utente interattiva.
- **OpenAI API**: Per utilizzare modelli linguistici avanzati e function calling. [Portale per le function calling di OpenAI](https://platform.openai.com/docs/guides/gpt/function-calling).
- **FlagEmbedding**: Per calcolare embeddings personalizzati con il modello `BGEM3FlagModel`. [Scopri il modello qui](https://github.com/FlagOpen/FlagEmbedding).
- **dotenv**: Per gestire le chiavi API in modo sicuro.

---

##
