from abc import ABC, abstractmethod

class FilterStrategy(ABC):
    """
    Interfaccia base per le strategie di filtraggio.
    """
    @abstractmethod
    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        pass

# --- FILTRI GENERICI ---

class UniversalFilter(FilterStrategy):
    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        if verbose:
            print(f"[FILTER] UniversalFilter: Accettato {sample.id}")
        return True

class TemporalFilter(FilterStrategy):
    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        if not sample.question:
            return False
            
        q = sample.question.lower().strip()
        time_keywords = ["when", "year", "date", "born", "died", "century", "how long", "period"]
        
        satisfied = any(keyword in q for keyword in time_keywords)
        if verbose and satisfied:
            print(f"[FILTER] TemporalFilter: Trovata keyword temporale in {sample.id}")
        return satisfied

class EntityFilter(FilterStrategy):
    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        if not sample.question:
            return False
            
        q = sample.question.lower().strip()
        satisfied = q.startswith("who") or q.startswith("which") or "name of" in q
        
        if verbose and satisfied:
            print(f"[FILTER] EntityFilter: Domanda su entità rilevata per {sample.id}")
        return satisfied

# --- FILTRI SPECIFICI PER DATASET ---

class HotpotBridgeFilter(FilterStrategy):
    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        is_bridge = getattr(sample, 'type', None) == "bridge"
        has_keys = len(sample.key_sentences) > 0
        
        satisfied = is_bridge and has_keys
        
        if verbose and not satisfied:
            reason = "non è bridge" if not is_bridge else "mancano key_sentences"
            print(f"[FILTER] HotpotBridgeFilter scartato {sample.id}: {reason}")
            
        return satisfied

class TruthfulQAFilter(FilterStrategy):
    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        satisfied = getattr(sample, 'type', None) == "truthful_qa"
        if verbose and satisfied:
            print(f"[FILTER] TruthfulQAFilter: Campione {sample.id} accettato.")
        return satisfied

# --- FILTRI COMPOSITI ---

class AndFilter(FilterStrategy):
    def __init__(self, filters: list[FilterStrategy]):
        self.filters = filters

    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        # Passiamo il verbose ai sotto-filtri
        res = all(f.is_satisfied(sample, verbose=verbose) for f in self.filters)
        if verbose:
            print(f"[FILTER-LOGIC] AND result per {sample.id}: {res}")
        return res

class OrFilter(FilterStrategy):
    def __init__(self, filters: list[FilterStrategy]):
        self.filters = filters

    def is_satisfied(self, sample, verbose: bool = False) -> bool:
        res = any(f.is_satisfied(sample, verbose=verbose) for f in self.filters)
        if verbose:
            print(f"[FILTER-LOGIC] OR result per {sample.id}: {res}")
        return res