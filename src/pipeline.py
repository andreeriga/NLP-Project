import pandas as pd
from .models import LLMModel
from .perturbation import Perturbator
from .verifier import BaseVerifier

class ExperimentPipeline:
    def __init__(self, model: LLMModel, perturbator: Perturbator, verifier: BaseVerifier = None, verbose: bool = False):
        self.model = model
        self.perturbator = perturbator
        self.verifier = verifier
        self.verbose = verbose
        self.results = []
        
        if self.verbose:
            print(f"[INFO] Pipeline inizializzata. Verificatore presente: {self.verifier is not None}")

    def run(self, samples, verbose: bool = None):
        is_verbose = verbose if verbose is not None else self.verbose
        
        total = len(samples)
        for i, s in enumerate(samples):
            if is_verbose:
                print(f"\n" + "="*50)
                print(f"[PIPELINE] Elaborazione Campione {i+1}/{total} (ID: {s.id})")
                print("="*50)

            # 1. Baseline Run
            if is_verbose: print(f"\n--- MODALITÀ: BASELINE ---")
            self._run_single_test(s, s.full_context, "baseline", verbose=is_verbose)
            
            # 2. Adversarial Run
            if is_verbose: print(f"\n--- MODALITÀ: ADVERSARIAL ---")
            adv_context = self.perturbator.create_adversarial_context(s, verbose=is_verbose)
            self._run_single_test(s, adv_context, "adversarial", verbose=is_verbose)

    def _run_single_test(self, sample, context, setup_type, verbose: bool = False):
        # Prompt CoT standard
        prompt = (
            f"Context: {context}\n"
            f"Question: {sample.question}\n"
            f"Answer based ONLY on context. Let's think step by step:"
        )
        
        if verbose:
            print(f"[INFO] Esecuzione inferenza ({setup_type})...")

        output = self.model.generate_response(prompt)
        
        # Fase di Verifica (opzionale)
        revised = None
        if self.verifier:
            if verbose:
                print(f"[INFO] Avvio fase di verifica/correzione...")
            ver_res = self.verifier.verify(context, sample.question, output, verbose=verbose)
            revised = ver_res['revised']

        self.results.append({
            "id": sample.id,
            "question": sample.question,
            "setup": setup_type,
            "output": output,
            "revised_output": revised,
            "ground_truth": sample.ground_truth
        })

    def save_results(self, filename="results.csv"):
        if self.verbose:
            print(f"\n[INFO] Salvataggio di {len(self.results)} risultati in {filename}...")
        pd.DataFrame(self.results).to_csv(filename, index=False)
        if self.verbose:
            print("[INFO] Salvataggio completato.")