from typing import List, Dict
from .metrics import MetricStrategy

class Evaluator:
    def __init__(self, metrics: List[MetricStrategy]):
        self.metrics = metrics

    def evaluate_all(self, prediction: str, truth: str) -> Dict[str, float]:
        """
        Esegue tutte le metriche configurate e ritorna un dizionario.
        Es: {'EM': 1, 'F1': 0.85}
        """
        results = {}
        for metric in self.metrics:
            # Calcola e salva usando il nome della metrica come chiave
            results[metric.name] = metric.compute(prediction, truth)
        return results