import re
import random

class Perturbator:
    def __init__(self, model):
        self.model = model

    def create_adversarial_context(self, sample, verbose=False):
        """
        Restituisce il contesto avversario.
        
        1. Controlla se esiste già un contesto generato da Gemini tramite api (High Quality).
        2. Se no, prova a generarlo con il modello locale.
        3. Se il modello locale fallisce, usa Regex (approccio forzato, preferirei non usarlo successivamente).
        4. Se il .replace() fallisce, forza l'iniezione della bugia all'inizio del testo.
        """
        
        # --- 1. USO DATI PRE-CALCOLATI (GEMINI) ---
        # Se il loader ha caricato il contesto avverso dal JSON, usiamo quello.
        if sample.adversarial_context:
            if verbose:
                print(f"[PERTURBATOR] Usato contesto pre-calcolato (Gemini) per {sample.id}")
            return sample.adversarial_context

        # --- 2. FALLBACK: GENERAZIONE LOCALE ---
        if not sample.key_sentences:
            return sample.full_context
            
        # Scegliamo la prima frase chiave come bersaglio
        target_fact = sample.key_sentences[0].strip()
        
        # Prompt per il modello locale
        prompt = (
            f"Rewrite the following sentence by changing ONLY the dates or names to make it factually wrong: "
            f"'{target_fact}'\n"
            f"False version:"
        )
        
        response = self.model.generate_response(prompt)
        fake_fact = response["text"].strip() if isinstance(response, dict) else response.strip()

        # --- 3. SAFETY NET (REGEX) ---
        # Se l'LLM locale ha fallito (copiato il testo o output vuoto)
        if fake_fact.lower() == target_fact.lower() or len(fake_fact) < 5:
            if verbose: print(f"[DEBUG] Perturbator LLM fallito per {sample.id}. Uso Regex.")
            
            # Sostituiamo gli anni (es. 1943 -> 1999)
            fake_fact = re.sub(r'\b(19|20)\d{2}\b', "1999", target_fact)
            
            # Sostituiamo i numeri generici (es. "seven Tony Awards" -> "99 Tony Awards")
            if fake_fact == target_fact:
                fake_fact = re.sub(r'\b\d+\b', "99", target_fact)
            
            # Se ancora non è cambiato nulla, usiamo la negazione esplicita
            if fake_fact == target_fact:
                fake_fact = f"It is false that {target_fact}"

        # --- 4. SOSTITUZIONE ROBUSTA (INIEZIONE) ---
        # Proviamo a sostituire la frase nel contesto
        if target_fact in sample.full_context:
            new_context = sample.full_context.replace(target_fact, fake_fact)
        else:
            # Se la frase esatta non viene trovata (colpa di spazi/newline diversi),
            # INIETTIAMO la bugia all'inizio del contesto come una "Correzione".
            # Questo assicura che il test avversario avvenga comunque.
            if verbose: 
                print(f"[WARNING] Target non trovato per replace in {sample.id}. Eseguo iniezione forzata.")
            
            new_context = f"[CORRECTION: {fake_fact}] " + sample.full_context

        return new_context