from abc import ABC, abstractmethod
import string
import collections

class MetricStrategy(ABC):
    """Interfaccia base per tutte le metriche di valutazione."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Il nome della metrica (usato come colonna nel CSV)."""
        pass

    @abstractmethod
    def compute(self, prediction: str, ground_truth: str) -> float:
        """Calcola il punteggio tra la predizione e la verità."""
        pass

    def normalize_answer(self, s):
        """Metodo di utilità condiviso per pulire il testo."""
        def remove_articles(text):
            return " ".join([w for w in text.split() if w not in ["a", "an", "the"]])

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

# --- IMPLEMENTAZIONI CONCRETE ---

class ExactMatchMetric(MetricStrategy):
    @property
    def name(self):
        return "EM"

    def compute(self, prediction: str, ground_truth: str) -> float:
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

class F1ScoreMetric(MetricStrategy):
    @property
    def name(self):
        return "F1"

    def compute(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common_tokens.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

class LengthRatioMetric(MetricStrategy):
    """
    (Bonus) Una metrica semplice che controlla se la risposta è
    troppo lunga o troppo corta rispetto alla verità.
    """
    @property
    def name(self):
        return "LenRatio"

    def compute(self, prediction: str, ground_truth: str) -> float:
        len_pred = len(prediction.split())
        len_true = len(ground_truth.split())
        if len_true == 0: return 0.0
        # Ritorna il rapporto (1.0 = lunghezza perfetta)
        return min(len_pred, len_true) / max(len_pred, len_true)