import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from .filters import FilterStrategy

# --- 1. STRUTTURA DATI UNIVERSALE ---
@dataclass
class ReasoningSample:
    id: str
    question: str
    ground_truth: str
    full_context: str
    key_sentences: List[str]
    type: str
    # CAMPO NUOVO: Contiene il testo avversario pre-calcolato (da Gemini), se esiste
    adversarial_context: Optional[str] = None 

# --- 2. CLASSE ASTRATTA ---
class BaseLoader(ABC):
    def __init__(self, file_path: str, verbose: bool = False):
        self.file_path = file_path
        self.verbose = verbose
        if self.verbose:
            print(f"[INFO] Inizializzazione {self.__class__.__name__} per: {self.file_path}")

    @abstractmethod
    def _load_raw_data(self):
        pass

    @abstractmethod
    def _parse_item(self, item) -> ReasoningSample:
        pass

    def load_filtered_data(self, strategy: FilterStrategy = None, limit: int = 50, verbose: bool = None) -> List[ReasoningSample]:
        # Override locale del verbose
        is_verbose = verbose if verbose is not None else self.verbose
        
        if is_verbose:
            print(f"[INFO] Caricamento dati da {self.file_path}...")
            if strategy:
                print(f"[INFO] Applicazione strategia di filtro: {type(strategy).__name__}")
        
        try:
            raw_data = self._load_raw_data()
        except FileNotFoundError:
            print(f"[ERROR] File non trovato: {self.file_path}")
            return []

        samples = []
        
        for item in raw_data:
            sample = self._parse_item(item)
            
            if strategy is None or strategy.is_satisfied(sample):
                samples.append(sample)
                if is_verbose and len(samples) % 10 == 0: # Log ogni 10 campioni
                    print(f"[INFO] Campioni raccolti: {len(samples)}/{limit}")
                
                if len(samples) >= limit:
                    break
        
        if is_verbose:
            print(f"[INFO] Caricamento completato. Totale: {len(samples)} campioni.")
        
        return samples

# --- 3. IMPLEMENTAZIONE HOTPOTQA ---
class HotpotLoader(BaseLoader):
    def _load_raw_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _format_context(self, context_list):
        """
        Helper per convertire la lista di liste di HotpotQA in stringa unica.
        Gestisce sia il contesto originale che quello adversarial di Gemini.
        """
        if not context_list:
            return None
        
        # Se è già stringa, ritornala
        if isinstance(context_list, str):
            return context_list

        text_parts = []
        # Struttura attesa: [['Title', ['sent1', 'sent2']], ...]
        for item in context_list:
            if len(item) >= 2:
                title = item[0]
                sentences = item[1]
                # Uniamo le frasi e aggiungiamo il titolo per chiarezza
                paragraph = f"Title: {title}. " + " ".join(sentences)
                text_parts.append(paragraph)
        
        return " ".join(text_parts)

    def _parse_item(self, entry) -> ReasoningSample:
        # Estrazione supporting facts per identificare le frasi chiave
        supporting_facts = entry.get('supporting_facts', [])
        key_sents = []
        
        # Logica per estrarre le frasi chiave specifiche dal contesto originale
        for title, sentences in entry['context']:
            for idx, sent in enumerate(sentences):
                # Se la frase è nei supporting facts, la salviamo
                if [title, idx] in supporting_facts:
                    key_sents.append(sent)
        
        # 1. Formattiamo il contesto originale
        original_text = self._format_context(entry['context'])

        # 2. Formattiamo il contesto adversarial (se presente nel JSON di Gemini)
        # Se il campo non esiste, adv_text sarà None
        adv_raw = entry.get('adversarial_context', None)
        adv_text = self._format_context(adv_raw)

        return ReasoningSample(
            id=entry['_id'],
            question=entry['question'],
            ground_truth=entry['answer'],
            full_context=original_text,
            key_sentences=key_sents,
            type=entry.get('type', 'unknown'),
            adversarial_context=adv_text # Passiamo il dato pre-calcolato
        )

# --- 4. IMPLEMENTAZIONE TRUTHFULQA ---
class TruthfulQALoader(BaseLoader):
    def _load_raw_data(self):
        return pd.read_csv(self.file_path).to_dict('records')

    def _parse_item(self, row) -> ReasoningSample:
        truth = row.get('Best Answer') or row.get('Correct Answers')
        
        return ReasoningSample(
            id=str(row.get('Question_ID', hash(row['Question']))),
            question=row['Question'],
            ground_truth=truth,
            full_context=truth, 
            key_sentences=[truth],
            type="truthful_qa",
            adversarial_context=None # TruthfulQA per ora non ha adversarial pre-calcolato
        )