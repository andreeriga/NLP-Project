import json
import time
import os
from typing import List, Dict, Optional
from google import genai
from google.genai import types

# --- CONFIGURAZIONE UTENTE ---
API_KEY = os.getenv("GEMINI_API_KEY", "QUI_CI_ANDREBBE_LA_CHIAVE") 

MODEL_NAME = "gemini-1.5-flash"
# Nota: i percorsi sono relativi alla cartella 'scripts/'
INPUT_FILE = "../data/HotpotQA_distractor.json" 
OUTPUT_FILE = "../data/hotpot_adversarial_gemini.json"
RATE_LIMIT_SLEEP = 4  # Secondi di pausa per evitare errore 429 (Too Many Requests)

class AdversarialGenerator:
    """
    Classe per gestire la generazione di contesti avversari tramite API Gemini.
    """

    def __init__(self, api_key: str, model_name: str):
        if not api_key or "INCOLLA" in api_key:
            raise ValueError("Errore: API Key mancante o non valida.")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Configurazione Safety: disabilitiamo i blocchi per permettere
        # la generazione di "falsità" necessarie per il test.
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        ]

    def generate_adversarial_entry(self, entry: Dict) -> Optional[List]:
        """
        Invia una richiesta a Gemini per modificare il contesto di un singolo campione.
        """
        json_str = json.dumps(entry)
        
        prompt = f"""
        You are an expert Red Teaming Assistant specialized in NLP robustness.
        Task: Modify the 'context' field of this HotpotQA entry to contradict the ground truth answer.
        
        INSTRUCTIONS:
        1. Analyze the 'question' and 'answer'.
        2. Identify the specific sentences in the 'context' that support the answer.
        3. Rewrite ONLY those sentences to make them FACTUALLY WRONG (e.g., change dates, names, invert boolean logic).
        4. Keep the exact Python list-of-lists structure: [['Title', ['Sent1', 'Sent2']]].
        5. Do NOT change the meaning of irrelevant sentences.

        INPUT ENTRY:
        {json_str}

        OUTPUT:
        The modified context list (JSON format only).
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                    safety_settings=self.safety_settings
                )
            )
            
            # Parsing della risposta
            return json.loads(response.text)

        except Exception as e:
            print(f"[ERROR] Errore generazione per ID {entry.get('_id')}: {e}")
            return None

    def process_dataset(self, input_path: str, output_path: str, limit: int = 50):
        """
        Carica il dataset, genera le perturbazioni e salva il risultato.
        """
        print(f"[INFO] Caricamento dataset da: {input_path}")
        
        try:
            with open(input_path, "r", encoding='utf-8') as f:
                original_data = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] File non trovato: {input_path}")
            print(f"[HINT] Assicurati di lanciare lo script dalla cartella 'scripts' o correggi il percorso.")
            return

        # Selezione sottogruppo di dati
        target_samples = original_data[:limit]
        new_dataset = []
        
        print(f"[INFO] Inizio generazione per {len(target_samples)} campioni con modello {self.model_name}...")
        print("-" * 50)

        for i, entry in enumerate(target_samples):
            print(f"[INFO] Processing [{i+1}/{len(target_samples)}] - ID: {entry['_id']}")
            
            adv_context = self.generate_adversarial_entry(entry)
            
            if adv_context:
                new_entry = entry.copy()
                new_entry['adversarial_context'] = adv_context
                new_entry['adversarial_model_source'] = self.model_name
                new_dataset.append(new_entry)
            else:
                print(f"[WARNING] Skipping ID {entry['_id']} a causa di un errore API.")

            # Pausa per rispettare i limiti del piano Free (circa 15 RPM)
            time.sleep(RATE_LIMIT_SLEEP)

        print("-" * 50)
        
        # Salvataggio su file
        try:
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(new_dataset, f, indent=4)
            print(f"[SUCCESS] Salvati {len(new_dataset)} campioni in: {output_path}")
        except Exception as e:
            print(f"[ERROR] Impossibile salvare il file: {e}")

if __name__ == "__main__":
    try:
        generator = AdversarialGenerator(API_KEY, MODEL_NAME)
        generator.process_dataset(INPUT_FILE, OUTPUT_FILE, limit=50)
        
    except KeyboardInterrupt:
        print("\n[INFO] Interruzione manuale dell'utente.")
    except Exception as e:
        print(f"\n[CRITICAL] Errore critico durante l'esecuzione: {e}")