import random
from .models import LLMModel

class Perturbator:
    def __init__(self, model: LLMModel, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        
        if self.verbose:
            print(f"[INFO] Perturbator inizializzato con modello: {type(model).__name__}")

    def create_adversarial_context(self, sample, verbose: bool = None) -> str:
        """Sostituisce una frase vera con una falsa generata dal modello."""
        is_verbose = verbose if verbose is not None else self.verbose
        
        if not sample.key_sentences:
            if is_verbose:
                print(f"[WARNING] Nessuna key_sentence trovata per il campione {sample.id}. Ritorno contesto originale.")
            return sample.full_context
            
        target_fact = random.choice(sample.key_sentences)
        
        # Prompt per generare la bugia
        lie_prompt = (
            f"Original sentence: {target_fact}\n"
            f"Rewrite this sentence to state the opposite or change the key entity/date to something plausible but false.\n"
            f"False sentence:"
        )
        
        if is_verbose:
            print(f"[INFO] Generazione perturbazione per il campione: {sample.id}")
            print(f"[INFO] Frase target selezionata: {target_fact}")

        fake_fact = self.model.generate_response(lie_prompt).strip()
        
        if is_verbose:
            print(f"[INFO] Frase perturbata (fake): {fake_fact}")
        
        # Sostituzione (semplice replace)
        adversarial_context = sample.full_context.replace(target_fact, fake_fact)
        
        if is_verbose:
            print("[INFO] Contesto adversarial creato con successo.")
            
        return adversarial_context