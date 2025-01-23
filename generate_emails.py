

# Importazione del modulo OpenAI per interagire con l'API OpenAI
import openai
import os
from dotenv import load_dotenv

# Carica le variabili di ambiente definite nel file .env
load_dotenv()

# Recupera la chiave API di OpenAI dall'ambiente
openai.api_key = os.environ['OpenAI_API_Key']


# Esegui un ciclo per creare 100 email
for i in range(10):
    # Stampa l'indice corrente per monitorare il progresso
    print(i)
    
    # Richiesta al modello GPT-4 per generare un'email realistica
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Specifica il modello da utilizzare
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Generi email realistiche di contatto, di preventivi, di informazioni, "
                            "di collaborazioni, di acquisto prodotti e tante altre, ricche di dettagli. "
                            "Inventati il cliente, il settore, il progetto che vogliono fare tutto rendendo "
                            "l'idea che sono email realistiche, metti anche cifre di preventivi e contratti "
                            "da firmare/firmati. Io mi chiamo Simone Rizzo e sono il CEO di Inferentia e "
                            "sviluppiamo soluzioni AI custom per i clienti su qualsiasi settore. Devi formattare "
                            "la mail in JSON."
                        )
                    }
                ]
            },
            {
                "role": "user",  # Contenuto fornito dall'utente
                "content": [
                    {
                        "type": "text",
                        "text": "genera una mail"
                    }
                ]
            },
            {
                "role": "assistant",  # Risposta attesa dall'assistente
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "{\"date\":\"25 ottobre 2023, 14:37 CET\","
                            "\"subject\":\"Richiesta di Preventivo per Servizi di Marketing Digitale\","
                            "\"from\":\"m.rossi@azienda.it\","
                            "\"body\":\"Gentile Sig.ra Bianchi,\\n\\nSpero che questa e-mail vi trovi bene. "
                            "Mi chiamo Marco Rossi e sono il direttore marketing di Azienda S.R.L. ...\"}"
                        )
                    }
                ]
            }
        ],
        response_format={  # Specifica il formato del risultato
            "type": "json_schema",  # Richiede un JSON basato su uno schema
            "json_schema": {
                "name": "email_schema",  # Nome dello schema
                "strict": True,  # Validazione rigida
                "schema": {  # Struttura dello schema JSON
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "La data e l'ora in cui l'email è stata inviata."
                        },
                        "subject": {
                            "type": "string",
                            "description": "L'oggetto dell'email."
                        },
                        "from": {
                            "type": "string",
                            "description": "L'indirizzo email del mittente."
                        },
                        "body": {
                            "type": "string",
                            "description": "Il contenuto principale o il corpo dell'email."
                        }
                    },
                    "required": [
                        "date",
                        "subject",
                        "from",
                        "body"
                    ],  # Campi obbligatori
                    "additionalProperties": False  # Impedisce campi extra non definiti nello schema
                }
            }
        },
        temperature=1,  # Controlla la casualità delle risposte
        max_completion_tokens=2048,  # Numero massimo di token generati
        top_p=1,  # Nucleo di campionamento
        frequency_penalty=0,  # Penalità per ripetizioni
        presence_penalty=0  # Penalità per l'assenza di nuovi argomenti
    )

    # Importa il modulo JSON per gestire le conversioni JSON in Python
    import json

    # Estrai il contenuto JSON dalla risposta del modello
    json_string = response.choices[0].message.content
    
    # Converte la stringa JSON in un dizionario Python
    email_data = json.loads(json_string)


    # Salva i dati email in un file JSON con un nome incrementale
    with open(f'data/email_{i}.json', 'w', encoding='utf-8') as f:
        json.dump(email_data, f, ensure_ascii=False, indent=4)
