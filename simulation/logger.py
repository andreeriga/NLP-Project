from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from simulation.runner import RunResult

logger = logging.getLogger(__name__)

TURNS_CSV_FIELDS = [
    "scenario_id", "run_id", "round_number", "speaker", "speaker_name",
    "utterance", "sentiment_label", "sentiment_score", "sentiment_compound",
    "hedge_rate", "question_rate", "avg_sentence_length", "lexical_diversity",
    "dominant_tactic", "persuasion_concession", "persuasion_flattery",
    "persuasion_threat", "persuasion_urgency", "named_outcomes",
]

RUNS_CSV_FIELDS = [
    "run_id", "scenario_id", "model_name", "timestamp", "elapsed_seconds",
    "total_rounds", "termination_reason", "outcome",
    "agent_a_utility", "agent_b_utility",
    "agent_a_above_batna", "agent_b_above_batna",
    "llm_winner", "llm_strategy", "llm_evaluation",
]


class TurnLogger:
    def __init__(self, runs_dir: str | Path = "data/runs") -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_csv  = self.runs_dir / "runs.csv"
        self.turns_csv = self.runs_dir / "turns.csv"
        self._init_csv(self.runs_csv,  RUNS_CSV_FIELDS)
        self._init_csv(self.turns_csv, TURNS_CSV_FIELDS)

    def log(self, result: RunResult) -> Path:
        run_dir = self._run_dir(result)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(result, run_dir)
        self._append_runs_csv(result)
        self._append_turns_csv(result)

        logger.info("Logged run '%s' to %s", result.run_id, run_dir)
        return run_dir

    def load_run(self, scenario_id: str, run_id: str) -> dict:
        path = self.runs_dir / scenario_id / run_id / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_runs(self, scenario_id: str | None = None) -> list[dict]:
        return self._read_csv(self.runs_csv, filter_key="scenario_id", filter_val=scenario_id)

    def load_turns(
        self,
        scenario_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict]:
        rows = self._read_csv(self.turns_csv, filter_key="scenario_id", filter_val=scenario_id)
        if run_id:
            rows = [r for r in rows if r["run_id"] == run_id]
        return rows

    def _write_json(self, result: RunResult, run_dir: Path) -> None:
        payload = {
            "scenario_id": result.scenario_id,
            "run_id": result.run_id,
            "model_name": result.model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": result.elapsed_seconds,
            "termination": {
                "reason": result.termination.reason,
                "outcome": result.termination.outcome,
                "agreed_by": result.termination.agreed_by,
            },
            "turns": [
                {
                    "round": t.round_number,
                    "speaker": t.speaker,
                    "speaker_name": t.speaker_name,
                    "utterance": t.utterance,
                    "nlp_features": t.nlp_features,
                }
                for t in result.history
            ],
            "turn_features": result.turn_features,
            "judgement": (
                self._serialise_judgement(result.judgement)
                if result.judgement else None
            ),
        }
        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _init_csv(self, path: Path, fields: list[str]) -> None:
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_runs_csv(self, result: RunResult) -> None:
        j = result.judgement
        row = {
            "run_id":               result.run_id,
            "scenario_id":          result.scenario_id,
            "model_name":           result.model_name,
            "timestamp":            datetime.utcnow().isoformat(),
            "elapsed_seconds":      result.elapsed_seconds,
            "total_rounds":         len(result.history),
            "termination_reason":   result.termination.reason,
            "outcome":              result.termination.outcome,
            "agent_a_utility":      j.agent_a_utility if j else "",
            "agent_b_utility":      j.agent_b_utility if j else "",
            "agent_a_above_batna":  int(j.agent_a_above_batna) if j else "",
            "agent_b_above_batna":  int(j.agent_b_above_batna) if j else "",
            "llm_winner":           j.llm_winner if j else "",
            "llm_strategy":         j.llm_strategy_summary if j else "",
            "llm_evaluation":       j.llm_evaluation if j else "",
        }
        self._append_row(self.runs_csv, RUNS_CSV_FIELDS, row)

    def _append_turns_csv(self, result: RunResult) -> None:
        features_by_round = {f["round"]: f for f in result.turn_features}
        for turn in result.history:
            f = features_by_round.get(turn.round_number, {})
            tactics = f.get("persuasion_tactics", {})
            row = {
                "scenario_id":          result.scenario_id,
                "run_id":               result.run_id,
                "round_number":         turn.round_number,
                "speaker":              turn.speaker,
                "speaker_name":         turn.speaker_name,
                "utterance":            turn.utterance,
                "sentiment_label":      f.get("sentiment_label", ""),
                "sentiment_score":      f.get("sentiment_score", ""),
                "sentiment_compound":   f.get("sentiment_compound", ""),
                "hedge_rate":           f.get("hedge_rate", ""),
                "question_rate":        f.get("question_rate", ""),
                "avg_sentence_length":  f.get("avg_sentence_length", ""),
                "lexical_diversity":    f.get("lexical_diversity", ""),
                "dominant_tactic":      f.get("dominant_tactic", ""),
                "persuasion_concession":tactics.get("concession", 0),
                "persuasion_flattery":  tactics.get("flattery", 0),
                "persuasion_threat":    tactics.get("threat", 0),
                "persuasion_urgency":   tactics.get("urgency", 0),
                "named_outcomes":       json.dumps(f.get("named_outcomes_mentioned", [])),
            }
            self._append_row(self.turns_csv, TURNS_CSV_FIELDS, row)

    def _append_row(self, path: Path, fields: list[str], row: dict) -> None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writerow(row)

    def _read_csv(
        self,
        path: Path,
        filter_key: str | None = None,
        filter_val: str | None = None,
    ) -> list[dict]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if filter_key and filter_val:
            rows = [r for r in rows if r.get(filter_key) == filter_val]
        return rows

    def _run_dir(self, result: RunResult) -> Path:
        return self.runs_dir / result.scenario_id / result.run_id

    def _serialise_judgement(self, j) -> dict:
        return {
            "outcome":              j.outcome,
            "termination_reason":   j.termination_reason,
            "agent_a_utility":      j.agent_a_utility,
            "agent_b_utility":      j.agent_b_utility,
            "agent_a_above_batna":  j.agent_a_above_batna,
            "agent_b_above_batna":  j.agent_b_above_batna,
            "total_rounds":         j.total_rounds,
            "dominant_tactics":     j.dominant_tactics,
            "sentiment_arcs":       j.sentiment_arcs,
            "avg_hedge_rate":       j.avg_hedge_rate,
            "avg_question_rate":    j.avg_question_rate,
            "llm_winner":           j.llm_winner,
            "llm_strategy_summary": j.llm_strategy_summary,
            "llm_evaluation":       j.llm_evaluation,
        }