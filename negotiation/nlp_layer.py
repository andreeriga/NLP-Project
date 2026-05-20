from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

import spacy
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "seem", "appears",
    "i think", "i believe", "i suppose", "kind of", "sort of", "somewhat",
    "probably", "generally", "usually", "often", "sometimes", "roughly",
}

PERSUASION_MARKERS = {
    "concession": ["however", "although", "even though", "despite", "but", "yet",
                   "i understand", "i see your point", "fair enough", "granted"],
    "flattery":   ["brilliant", "great idea", "excellent", "love that", "fantastic",
                   "you're right", "absolutely", "perfect", "genius", "impressive"],
    "threat":     ["otherwise", "or else", "i'll have to", "forced to", "no choice",
                   "unless", "if you don't", "walk away", "call it off"],
    "urgency":    ["now", "immediately", "right away", "tonight", "today",
                   "last chance", "final offer", "running out"],
}

@dataclass
class TurnFeatures:
    sentiment_label: str           
    sentiment_score: float         
    sentiment_compound: float      
    hedge_rate: float              
    question_rate: float           
    avg_sentence_length: float     
    lexical_diversity: float       
    persuasion_tactics: dict       
    named_outcomes_mentioned: list[str]  
    dominant_tactic: str           
    intent_label: str              # NUOVO: 'agreement', 'impasse', o 'ongoing'
    intent_score: float            # NUOVO: confidenza dell'intento

class NLPLayer:
    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        intent_model: str = "valhalla/distilbart-mnli-12-3", # Modello NLI leggero e veloce
        spacy_model: str = "en_core_web_sm",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_id = 0 if self.device == "cuda" else -1

        logger.info("Loading sentiment model: %s", sentiment_model)
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=sentiment_model,
            device=device_id,
            truncation=True,
            max_length=512,
        )

        logger.info("Loading zero-shot intent model: %s", intent_model)
        self.intent_pipeline = pipeline(
            "zero-shot-classification",
            model=intent_model,
            device=device_id,
        )

        logger.info("Loading spaCy model: %s", spacy_model)
        self.nlp = spacy.load(spacy_model)

    def analyse(self, text: str, possible_outcomes: list[str] | None = None) -> TurnFeatures:
        sentiment_label, sentiment_score, sentiment_compound = self._sentiment(text)
        intent_label, intent_score = self._detect_intent(text)

        doc = self.nlp(text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        sentences = list(doc.sents)

        hedge_rate = self._hedge_rate(text, tokens)
        question_rate = self._question_rate(sentences)
        avg_sentence_length = self._avg_sentence_length(sentences)
        lexical_diversity = self._lexical_diversity(tokens)
        persuasion_tactics = self._persuasion_tactics(text)
        named_outcomes = self._named_outcomes(text, possible_outcomes or [])

        counts = {k: v for k, v in persuasion_tactics.items() if v > 0}
        dominant_tactic = max(counts, key=counts.get) if counts else "none"

        return TurnFeatures(
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            sentiment_compound=sentiment_compound,
            hedge_rate=hedge_rate,
            question_rate=question_rate,
            avg_sentence_length=avg_sentence_length,
            lexical_diversity=lexical_diversity,
            persuasion_tactics=persuasion_tactics,
            named_outcomes_mentioned=named_outcomes,
            dominant_tactic=dominant_tactic,
            intent_label=intent_label,
            intent_score=intent_score,
        )

    def to_dict(self, features: TurnFeatures) -> dict:
        return {
            "sentiment_label": features.sentiment_label,
            "sentiment_score": round(features.sentiment_score, 4),
            "sentiment_compound": round(features.sentiment_compound, 4),
            "hedge_rate": round(features.hedge_rate, 4),
            "question_rate": round(features.question_rate, 4),
            "avg_sentence_length": round(features.avg_sentence_length, 2),
            "lexical_diversity": round(features.lexical_diversity, 4),
            "persuasion_tactics": features.persuasion_tactics,
            "named_outcomes_mentioned": features.named_outcomes_mentioned,
            "dominant_tactic": features.dominant_tactic,
            "intent_label": features.intent_label,
            "intent_score": round(features.intent_score, 4),
        }

    # --- Metodi Privati ---

    def _detect_intent(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return "ongoing", 0.0
            
        candidate_labels = ["reaching an agreement", "giving up or walking away", "negotiating or discussing"]
        result = self.intent_pipeline(text, candidate_labels=candidate_labels)
        
        best_match = result["labels"][0]
        score = result["scores"][0]
        
        if best_match == "reaching an agreement":
            return "agreement", score
        elif best_match == "giving up or walking away":
            return "impasse", score
        else:
            return "ongoing", score

    def _sentiment(self, text: str) -> tuple[str, float, float]:
        if not text.strip():
            return "neutral", 0.0, 0.0
        result = self.sentiment_pipeline(text)[0]
        raw_label = result["label"].lower()
        score = float(result["score"])
        
        if "pos" in raw_label:
            return "positive", score, score
        elif "neg" in raw_label:
            return "negative", score, -score
        else:
            return "neutral", score, 0.0

    def _hedge_rate(self, text: str, tokens: list) -> float:
        if not tokens: return 0.0
        text_lower = text.lower()
        count = sum(1 for hw in HEDGE_WORDS if hw in text_lower)
        return count / len(tokens)

    def _question_rate(self, sentences: list) -> float:
        if not sentences: return 0.0
        questions = sum(1 for s in sentences if s.text.strip().endswith("?"))
        return questions / len(sentences)

    def _avg_sentence_length(self, sentences: list) -> float:
        if not sentences: return 0.0
        lengths = [len([t for t in s if not t.is_punct and not t.is_space]) for s in sentences]
        return sum(lengths) / len(lengths)

    def _lexical_diversity(self, tokens: list) -> float:
        if not tokens: return 0.0
        words = [t.lower_ for t in tokens if t.is_alpha]
        if not words: return 0.0
        return len(set(words)) / len(words)

    def _persuasion_tactics(self, text: str) -> dict:
        text_lower = text.lower()
        return {
            tactic: sum(1 for marker in markers if marker in text_lower)
            for tactic, markers in PERSUASION_MARKERS.items()
        }

    def _named_outcomes(self, text: str, possible_outcomes: list[str]) -> list[str]:
        text_lower = text.lower()
        found = []
        for outcome in possible_outcomes:
            if "impasse" in outcome.lower(): continue
            clean = re.sub(r"\(.*?\)", "", outcome).strip().lower()
            keywords = [w for w in clean.split() if len(w) > 3]
            if any(kw in text_lower for kw in keywords):
                found.append(outcome)
        return found