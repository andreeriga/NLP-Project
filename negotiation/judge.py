from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from negotiation.scenario import Scenario, TerminationResult

logger = logging.getLogger(__name__)


@dataclass
class JudgementResult:
    scenario_id: str
    outcome: str | None
    termination_reason: str
    agent_a_utility: float
    agent_b_utility: float
    agent_a_above_batna: bool
    agent_b_above_batna: bool
    total_rounds: int
    dominant_tactics: dict
    sentiment_arcs: dict
    avg_hedge_rate: dict
    avg_question_rate: dict
    llm_evaluation: str
    llm_winner: str
    llm_strategy_summary: str


class JudgeAgent:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",  # Modello aggiornato
        max_new_tokens: int = 512,
        temperature: float = 0.1,  # Molto bassa per un output rigido e analitico
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = "auto" if torch.cuda.is_available() else "cpu"

        logger.info("Loading judge model: %s on %s", model_name, self.device_map)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # float16 per velocizzare l'inferenza e salvare VRAM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device_map != "cpu" else torch.float32,
            device_map=self.device_map,
            trust_remote_code=True,
        )

        # Rimosso device=-1 che forzava l'uso della CPU
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        logger.info("Judge model loaded.")

    def evaluate(
        self,
        scenario: Scenario,
        history: list,
        termination: TerminationResult,
        turn_features: list[dict],
    ) -> JudgementResult:
        outcome = termination.outcome
        reason  = termination.reason

        agent_a_utility     = scenario.utility_score("agent_a", outcome) if outcome else 0.0
        agent_b_utility     = scenario.utility_score("agent_b", outcome) if outcome else 0.0
        agent_a_above_batna = scenario.is_above_batna("agent_a", outcome) if outcome else False
        agent_b_above_batna = scenario.is_above_batna("agent_b", outcome) if outcome else False

        a_features = [f for f in turn_features if f.get("speaker") == "agent_a"]
        b_features = [f for f in turn_features if f.get("speaker") == "agent_b"]

        dominant_tactics = {
            "agent_a": self._dominant_tactic(a_features),
            "agent_b": self._dominant_tactic(b_features),
        }
        sentiment_arcs = {
            "agent_a": [f["sentiment_compound"] for f in a_features],
            "agent_b": [f["sentiment_compound"] for f in b_features],
        }
        avg_hedge = {
            "agent_a": self._avg(a_features, "hedge_rate"),
            "agent_b": self._avg(b_features, "hedge_rate"),
        }
        avg_question = {
            "agent_a": self._avg(a_features, "question_rate"),
            "agent_b": self._avg(b_features, "question_rate"),
        }

        llm_eval, llm_winner, llm_strategy = self._llm_evaluate(
            scenario, history, termination, agent_a_utility, agent_b_utility
        )

        return JudgementResult(
            scenario_id=scenario.id,
            outcome=outcome,
            termination_reason=reason,
            agent_a_utility=agent_a_utility,
            agent_b_utility=agent_b_utility,
            agent_a_above_batna=agent_a_above_batna,
            agent_b_above_batna=agent_b_above_batna,
            total_rounds=len(history),
            dominant_tactics=dominant_tactics,
            sentiment_arcs=sentiment_arcs,
            avg_hedge_rate=avg_hedge,
            avg_question_rate=avg_question,
            llm_evaluation=llm_eval,
            llm_winner=llm_winner,
            llm_strategy_summary=llm_strategy,
        )

    def to_dict(self, result: JudgementResult) -> dict:
        return {
            "scenario_id":          result.scenario_id,
            "outcome":              result.outcome,
            "termination_reason":   result.termination_reason,
            "agent_a_utility":      result.agent_a_utility,
            "agent_b_utility":      result.agent_b_utility,
            "agent_a_above_batna":  result.agent_a_above_batna,
            "agent_b_above_batna":  result.agent_b_above_batna,
            "total_rounds":         result.total_rounds,
            "dominant_tactics":     result.dominant_tactics,
            "sentiment_arcs":       result.sentiment_arcs,
            "avg_hedge_rate":       result.avg_hedge_rate,
            "avg_question_rate":    result.avg_question_rate,
            "llm_evaluation":       result.llm_evaluation,
            "llm_winner":           result.llm_winner,
            "llm_strategy_summary": result.llm_strategy_summary,
        }

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _llm_evaluate(
        self,
        scenario: Scenario,
        history: list,
        termination: TerminationResult,
        utility_a: float,
        utility_b: float,
    ) -> tuple[str, str, str]:
        transcript = "\n".join(
            f"{t.speaker_name}: {t.utterance}" for t in history
        )

        snap_a = scenario.get_agent_snapshot("agent_a")
        snap_b = scenario.get_agent_snapshot("agent_b")

        system_msg = "You are a neutral, expert negotiation analyst. Your job is to evaluate the negotiation and strictly follow the requested output format."
        
        user_msg = (
            f"SCENARIO: {scenario.title}\n"
            f"GAME TYPE: {scenario.game_theory_type}\n\n"
            f"AGENT A ({snap_a.name}) — secret goal: '{snap_a.best_outcome}', BATNA: '{snap_a.batna}'\n"
            f"AGENT B ({snap_b.name}) — secret goal: '{snap_b.best_outcome}', BATNA: '{snap_b.batna}'\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"OUTCOME: {termination.outcome} (reason: {termination.reason})\n"
            f"UTILITY SCORES: {snap_a.name}={utility_a:.2f}, {snap_b.name}={utility_b:.2f}\n\n"
            f"Respond in exactly this format:\n"
            f"WINNER: <agent_a | agent_b | draw | impasse>\n"
            f"STRATEGY_A: <one sentence summary>\n"
            f"STRATEGY_B: <one sentence summary>\n"
            f"EVALUATION: <3-5 sentences evaluating the negotiation>\n"
        )

        # Usiamo il formato ChatML di Qwen
        prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        raw = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=False,  # disattiviamo il sampling per forzare determinismo
            return_full_text=False,
        )[0]["generated_text"].strip()

        # Rimuoviamo eventuali tag residui di fine turno
        raw = raw.replace("<|im_end|>", "").strip()

        winner     = self._parse_field(raw, "WINNER",     "impasse")
        strategy_a = self._parse_field(raw, "STRATEGY_A", "")
        strategy_b = self._parse_field(raw, "STRATEGY_B", "")
        evaluation = self._parse_field(raw, "EVALUATION", raw)

        strategy_summary = f"{snap_a.name}: {strategy_a} | {snap_b.name}: {strategy_b}"

        return evaluation, winner, strategy_summary

    def _parse_field(self, text: str, field: str, default: str) -> str:
        for line in text.splitlines():
            if line.upper().startswith(field + ":"):
                return line[len(field) + 1:].strip()
        return default

    def _dominant_tactic(self, features: list[dict]) -> str:
        totals: dict[str, int] = {}
        for f in features:
            for tactic, count in f.get("persuasion_tactics", {}).items():
                totals[tactic] = totals.get(tactic, 0) + count
        active = {k: v for k, v in totals.items() if v > 0}
        return max(active, key=active.get) if active else "none"

    def _avg(self, features: list[dict], key: str) -> float:
        values = [f[key] for f in features if key in f]
        return round(sum(values) / len(values), 4) if values else 0.0