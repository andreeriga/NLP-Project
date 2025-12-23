import pandas as pd
from .models import LLMModel
from .perturbation import Perturbator
from .verifier import BaseVerifier
from .explainability import AttentionProfiler
from .evaluator import Evaluator

class ExperimentPipeline:
    def __init__(self, model: LLMModel, perturbator: Perturbator, verifier: BaseVerifier, evaluator: Evaluator, verbose: bool = False):
        """
        Inizializza la pipeline con tutti i componenti (Dependency Injection).
        :param evaluator: Istanza di Evaluator configurata con le metriche desiderate.
        """
        self.model = model
        self.perturbator = perturbator
        self.verifier = verifier
        self.evaluator = evaluator # Gestore delle metriche
        self.verbose = verbose
        self.results = []
        
        # Inizializza il profiler per l'analisi dell'attenzione (se il modello è presente)
        self.profiler = AttentionProfiler(model) if model else None

    def run(self, samples, deep_analysis: bool = False, analysis_limit: int = 5):
        """
        Esegue l'esperimento su una lista di campioni.
        :param deep_analysis: Se True, calcola i punteggi di attenzione (lento).
        :param analysis_limit: Se deep_analysis è True, lo applica solo ai primi N campioni.
        """
        total = len(samples)
        
        for i, s in enumerate(samples):
            # LOGICA IBRIDA: Deep Analysis solo sui primi 'analysis_limit' campioni
            should_analyze = deep_analysis and (i < analysis_limit)

            if self.verbose:
                extra_log = " [+ DEEP ANALYSIS]" if should_analyze else ""
                print(f"\n[PIPELINE] Campione {i+1}/{total}{extra_log} - ID: {s.id}")

            # 1. Baseline Run (Contesto Originale)
            self._run_single_test(s, s.full_context, "baseline", should_analyze)
            
            # 2. Adversarial Run (Contesto Perturbato/Bugia)
            # Nota: il perturbator crea un contesto con informazioni false
            adv_context = self.perturbator.create_adversarial_context(s)
            self._run_single_test(s, adv_context, "adversarial", should_analyze)

    def _run_single_test(self, sample, context, setup_type, do_analysis):
        # 1. Costruzione Prompt Chain-of-Thought
        prompt = (
            f"Context: {context}\n"
            f"Question: {sample.question}\n"
            f"Answer based ONLY on context. Let's think step by step:"
        )
        
        # 2. Inferenza (Generazione)
        # Ritorna un dizionario con testo e confidenza
        res = self.model.generate_response(prompt)
        output_text = res["text"]
        confidence = res["confidence"]

        # 3. Deep Analysis (Attention Profiling) - Opzionale
        att_context = None
        if do_analysis and self.profiler:
            # Cerchiamo quanto il modello ha guardato la prima frase chiave (o "Context" come fallback)
            # Prendiamo solo i primi token per evitare errori di lunghezza
            target = sample.key_sentences[0][:20] if sample.key_sentences else "Context"
            att_context = self.profiler.get_token_attention_score(prompt, output_text, target)

        # 4. Fase di Verifica (Self-Correction)
        revised_output = None
        if self.verifier:
            ver_res = self.verifier.verify(context, sample.question, output_text)
            revised_output = ver_res['revised']

        # 5. Valutazione Metriche (Strategy Pattern)
        # Calcoliamo le metriche sulla prima risposta
        metrics_dict = self.evaluator.evaluate_all(output_text, sample.ground_truth)
        
        # Se esiste una revisione, calcoliamo le metriche anche su quella
        metrics_revised_dict = {}
        if revised_output:
            raw_metrics = self.evaluator.evaluate_all(revised_output, sample.ground_truth)
            # Aggiungiamo il prefisso "revised_" per distinguerle nel CSV
            metrics_revised_dict = {f"revised_{k}": v for k, v in raw_metrics.items()}

        # 6. Aggregazione Risultati
        result_entry = {
            "id": sample.id,
            "dataset_type": getattr(sample, 'type', 'unknown'),
            "setup": setup_type,
            "question": sample.question,
            "context": context,              # <--- AGGIUNGI QUESTA RIGA
            "ground_truth": sample.ground_truth,
            "output": output_text,
            "confidence": confidence,
            "revised_output": revised_output,
            "attention_context_score": att_context,
        }
        
        # Unione dei dizionari delle metriche
        result_entry.update(metrics_dict)          # Es: EM, F1
        result_entry.update(metrics_revised_dict)  # Es: revised_EM, revised_F1

        self.results.append(result_entry)

    def save_results(self, filename="results.csv"):
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            if self.verbose: print(f"[INFO] Risultati salvati correttamente in {filename}")
        else:
            print("[WARNING] Nessun risultato da salvare.")