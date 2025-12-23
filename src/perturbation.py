import re
import random

class Perturbator:
    def __init__(self, model):
        self.model = model

    def create_adversarial_context(self, sample, verbose=False):
        """
        Crea un contesto falso sostituendo una frase chiave con una bugia.
        Include un sistema di sicurezza se l'LLM fallisce.
        """
        if not sample.key_sentences:
            return sample.full_context
            
        # Scegliamo la prima frase chiave come bersaglio
        target_fact = sample.key_sentences[0]
        
        # 1. TENTATIVO CON LLM
        prompt = (
            f"Rewrite the following sentence by changing ONLY the dates or names to make it factually wrong: "
            f"'{target_fact}'\n"
            f"False version:"
        )
        
        response = self.model.generate_response(prompt)
        # Se il modello restituisce un dict (come nel nostro caso), prendiamo 'text'
        fake_fact = response["text"].strip() if isinstance(response, dict) else response.strip()

        # 2. LOGICA DI SICUREZZA (FALLBACK)
        # Se l'LLM ha fatto copy-paste del testo originale o ha dato una risposta troppo corta
        if fake_fact.lower() == target_fact.lower() or len(fake_fact) < 5:
            if verbose: print(f"[DEBUG] Perturbator LLM fallito per {sample.id}. Uso Regex.")
            
            # Sostituiamo gli anni (es. 1943 -> 1999)
            fake_fact = re.sub(r'\b(19|20)\d{2}\b', "1999", target_fact)
            
            # Sostituiamo i numeri generici (es. "seven Tony Awards" -> "99 Tony Awards")
            if fake_fact == target_fact:
                fake_fact = re.sub(r'\b\d+\b', "99", target_fact)
            
            # Se ancora non è cambiato nulla, usiamo la negazione
            if fake_fact == target_fact:
                fake_fact = f"It is false that {target_fact}"

        # 3. SOSTITUZIONE ROBUSTA
        # Usiamo replace ma con un piccolo check per assicurarci che avvenga
        new_context = sample.full_context.replace(target_fact, fake_fact)
        
        # Se per qualche motivo il replace non ha funzionato (es. spazi diversi)
        # forziamo l'inserimento in testa o in coda (opzionale)
        if new_context == sample.full_context and verbose:
            print(f"[WARNING] Sostituzione testuale fallita nel contesto per {sample.id}")

        return new_context