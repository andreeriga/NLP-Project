import os
from src.models import HuggingFaceModel
from src.data_loader import HotpotLoader, TruthfulQALoader
from src.filters import (
    HotpotBridgeFilter, 
    TruthfulQAFilter, 
    TemporalFilter, 
    AndFilter,
    LengthFilter
)
from src.perturbation import Perturbator
from src.verifier import SelfCorrectionVerifier
from src.metrics import ExactMatchMetric, F1ScoreMetric
from src.evaluator import Evaluator
from src.pipeline import ExperimentPipeline

# --- CONFIGURAZIONE GLOBALE ---
MODEL_NAME = "google/flan-t5-base"

# File Paths
# 1. File principale (Generato da Gemini con adversarial context)
HOTPOT_GEMINI_FILE = "data/hotpot_adversarial_gemini.json"
# 2. File di backup (Originale, se il primo non esiste)
HOTPOT_BACKUP_FILE = "data/hotpot_dev_distractor_v1.json" 
# 3. File TruthfulQA
TRUTHFUL_FILE = "data/TruthfulQA.csv"

OUTPUT_FILE = "project_results.csv"

# Parametri Esecuzione
VERBOSE = True
TOTAL_SAMPLES_TO_LOAD = 40  # Totale campioni da caricare (es. 20 + 20)
DEEP_ANALYSIS = True        # Attiva il calcolo dell'attenzione
ANALYSIS_LIMIT = 5          # Esegui deep analysis solo sui primi 5 campioni

def main():
    print(f"\n--- AVVIO PROGETTO: Truth, Lies & Reasoning Machines ---")
    print(f"Modello: {MODEL_NAME}")
    
    # ---------------------------------------------------------
    # 1. SETUP COMPONENTI (DEPENDENCY INJECTION)
    # ---------------------------------------------------------
    # Inizializzazione Modello
    llm = HuggingFaceModel(MODEL_NAME, verbose=VERBOSE)
    
    # Componenti Logici
    perturbator = Perturbator(model=llm)         # Generatore di bugie (o lettore da Gemini)
    verifier = SelfCorrectionVerifier(model=llm) # Modulo di auto-correzione
    
    # Configurazione Metriche (Strategy Pattern)
    active_metrics = [
        ExactMatchMetric(),
        F1ScoreMetric()
    ]
    evaluator_manager = Evaluator(metrics=active_metrics)

    # ---------------------------------------------------------
    # 2. CARICAMENTO DATI (MULTI-DATASET)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FASE 1: Caricamento e Filtraggio Dati")
    print("="*50)

    samples_hotpot = []
    samples_truthful = []
    limit_per_ds = TOTAL_SAMPLES_TO_LOAD // 2

    # Filtro lunghezza per evitare crash su T5 (max 400 token)
    len_filter = LengthFilter(max_tokens=400)

    # --- A. HotpotQA (Logica Ibrida Gemini/Backup) ---
    target_hotpot_file = HOTPOT_GEMINI_FILE
    
    # Check preliminare: Se il file Gemini non c'è, usiamo il backup
    if not os.path.exists(target_hotpot_file):
        print(f"[WARNING] File Gemini ({HOTPOT_GEMINI_FILE}) non trovato.")
        print(f"[INFO] Switch automatico al dataset originale: {HOTPOT_BACKUP_FILE}")
        target_hotpot_file = HOTPOT_BACKUP_FILE
    
    if os.path.exists(target_hotpot_file):
        print(f"[HotpotQA] Caricamento da: {target_hotpot_file}")
        
        # Istanziamo il loader con il file selezionato
        loader_hp = HotpotLoader(target_hotpot_file)
        
        # Filtro: Domande complesse (Bridge) E Temporali E Lunghezza
        strategy_hp = AndFilter([
            HotpotBridgeFilter(), 
            TemporalFilter(), 
            len_filter
        ])
        
        samples_hotpot = loader_hp.load_filtered_data(
            strategy=strategy_hp, 
            limit=limit_per_ds,
            verbose=VERBOSE
        )
    else:
        print(f"[ERROR] Nessun file HotpotQA trovato (né Gemini né originale).")

    # --- B. TruthfulQA ---
    if os.path.exists(TRUTHFUL_FILE):
        print(f"\n[TruthfulQA] Caricamento da: {TRUTHFUL_FILE}")
        loader_tqa = TruthfulQALoader(TRUTHFUL_FILE)
        
        # Filtro: Anche qui cerchiamo domande temporali per coerenza + lunghezza
        strategy_tqa = AndFilter([
            TruthfulQAFilter(), 
            TemporalFilter(), 
            len_filter
        ])
        
        samples_truthful = loader_tqa.load_filtered_data(
            strategy=strategy_tqa, 
            limit=limit_per_ds,
            verbose=VERBOSE
        )
    else:
        print(f"[WARNING] File {TRUTHFUL_FILE} non trovato.")

    # Unione
    all_samples = samples_hotpot + samples_truthful
    print(f"\n[INFO] Totale campioni pronti: {len(all_samples)}")

    if not all_samples:
        print("[ERROR] Nessun dato caricato. Interruzione.")
        return

    # ---------------------------------------------------------
    # 3. ESECUZIONE PIPELINE
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FASE 2: Esecuzione Esperimento")
    print("="*50)

    # Istanziamo la pipeline iniettando TUTTI i componenti necessari
    pipeline = ExperimentPipeline(
        model=llm,
        perturbator=perturbator,
        verifier=verifier,
        evaluator=evaluator_manager,
        verbose=VERBOSE
    )

    # Avvio Run Ibrida
    pipeline.run(
        samples=all_samples,
        deep_analysis=DEEP_ANALYSIS,  # Attiva il profilatore
        analysis_limit=ANALYSIS_LIMIT # Limita il profilatore ai primi N casi
    )

    # ---------------------------------------------------------
    # 4. SALVATAGGIO
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FASE 3: Output")
    print("="*50)
    
    pipeline.save_results(OUTPUT_FILE)
    print(f"Processo completato. Analizza i risultati in: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()