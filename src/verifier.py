from abc import ABC, abstractmethod
from .models import LLMModel

class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, context, question, answer, verbose: bool = None) -> dict:
        pass

class SelfCorrectionVerifier(BaseVerifier):
    def __init__(self, model: LLMModel, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        
        if self.verbose:
            print(f"[INFO] Verifier inizializzato con modello: {type(model).__name__}")

    def verify(self, context, question, answer, verbose: bool = None) -> dict:
        is_verbose = verbose if verbose is not None else self.verbose
        
        prompt = (
            f"Context: {context}\nQuestion: {question}\nProposed Answer: {answer}\n"
            f"Check if the answer is consistent with the Context. If wrong, correct it.\n"
            f"Correct Answer:"
        )
        
        if is_verbose:
            print(f"[INFO] Verifica della coerenza in corso...")
            print(f"[INFO] Risposta proposta da analizzare: '{answer}'")

        revised = self.model.generate_response(prompt)['text'].strip()
        has_changed = answer.strip() != revised
        
        if is_verbose:
            if has_changed:
                print(f"[INFO] Correzione applicata!")
                print(f"[INFO] Nuova risposta: '{revised}'")
            else:
                print(f"[INFO] La risposta è stata confermata come corretta.")
                
        return {
            "original": answer, 
            "revised": revised, 
            "changed": has_changed
        }