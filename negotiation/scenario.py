from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from negotiation.agent import AgentConfig

if TYPE_CHECKING:
    from negotiation.agent import Turn, SpeakerRole

logger = logging.getLogger(__name__)

@dataclass
class TerminationResult:
    terminated: bool
    reason: str
    outcome: str | None = None
    agreed_by: list[str] = field(default_factory=list)

@dataclass
class AgentSnapshot:
    name: str
    role: str          
    game_theory_type: str
    utility_ranking: list[str]
    batna: str
    best_outcome: str  
    worst_outcome: str 

class Scenario:
    def __init__(self, data: dict) -> None:
        self._data = data
        self._validate()

    def _validate(self) -> None:
        required_top = [
            "id", "title", "description",
            "game_theory_type", "possible_outcomes",
            "termination_conditions", "agent_a", "agent_b",
        ]
        for key in required_top:
            if key not in self._data:
                raise ValueError(f"Scenario '{self._data.get('id', '?')}' missing field: '{key}'")

        required_agent = [
            "name", "role_description", "public_goal",
            "hidden_agenda", "utility_ranking", "batna", "batna_notes",
        ]
        for role in ("agent_a", "agent_b"):
            agent = self._data[role]
            for key in required_agent:
                if key not in agent:
                    raise ValueError(f"Scenario '{self.id}' / '{role}' missing field: '{key}'")
            
            ranking = agent["utility_ranking"]
            outcomes = self.possible_outcomes
            for item in ranking:
                if item not in outcomes:
                    raise ValueError(f"Scenario '{self.id}' / '{role}': utility_ranking item '{item}' not found.")
            
            if agent["batna"] not in outcomes:
                raise ValueError(f"Scenario '{self.id}' / '{role}': BATNA '{agent['batna']}' not found.")

        tc = self._data["termination_conditions"]
        for key in ("max_rounds", "agreement_signal", "impasse_signal"):
            if key not in tc:
                raise ValueError(f"Scenario '{self.id}': termination_conditions missing '{key}'")

        logger.debug("Scenario '%s' validated successfully.", self.id)

    @property
    def id(self) -> str: return self._data["id"]
    @property
    def title(self) -> str: return self._data["title"]
    @property
    def description(self) -> str: return self._data["description"]
    @property
    def game_theory_type(self) -> str: return self._data["game_theory_type"]
    @property
    def game_theory_notes(self) -> str: return self._data.get("game_theory_notes", "")
    @property
    def possible_outcomes(self) -> list[str]: return self._data["possible_outcomes"]
    @property
    def max_rounds(self) -> int: return int(self._data["termination_conditions"]["max_rounds"])
    @property
    def agreement_signal(self) -> str: return self._data["termination_conditions"]["agreement_signal"]
    @property
    def impasse_signal(self) -> str: return self._data["termination_conditions"]["impasse_signal"]

    def get_agent_config(self, role: "SpeakerRole") -> "AgentConfig":
        agent = self._data[role]
        return AgentConfig(
            name=agent["name"],
            role_description=agent["role_description"],
            public_goal=agent["public_goal"],
            hidden_agenda=agent["hidden_agenda"],
            utility_ranking=agent["utility_ranking"],
            batna=agent["batna"],
            batna_notes=agent["batna_notes"],
            game_theory_type=self.game_theory_type,
        )

    def get_agent_snapshot(self, role: "SpeakerRole") -> AgentSnapshot:
        agent = self._data[role]
        ranking = agent["utility_ranking"]
        return AgentSnapshot(
            name=agent["name"],
            role=role,
            game_theory_type=self.game_theory_type,
            utility_ranking=ranking,
            batna=agent["batna"],
            best_outcome=ranking[0],
            worst_outcome=ranking[-1],
        )

    def utility_score(self, role: "SpeakerRole", outcome: str) -> float:
        ranking = self._data[role]["utility_ranking"]
        n = len(ranking)
        if n <= 1: return 1.0
        try:
            rank = ranking.index(outcome)
        except ValueError:
            return -1.0
        return round((n - 1 - rank) / (n - 1), 4)

    def batna_score(self, role: "SpeakerRole") -> float:
        return self.utility_score(role, self._data[role]["batna"])

    def is_above_batna(self, role: "SpeakerRole", outcome: str) -> bool:
        return self.utility_score(role, outcome) > self.batna_score(role)

    def check_termination(
        self,
        history: list["Turn"],
        current_round: int,
        latest_features: dict = None,
    ) -> TerminationResult:
        
        # 1. Controllo max rounds
        if current_round >= self.max_rounds:
            logger.info("Scenario '%s': max rounds (%d) reached.", self.id, self.max_rounds)
            agreed_outcome = self._detect_agreement(history)
            if agreed_outcome:
                names = [t.speaker_name for t in history[-2:]]
                return TerminationResult(terminated=True, reason="agreement", outcome=agreed_outcome, agreed_by=names)
            return TerminationResult(
                terminated=True,
                reason="max_rounds",
                outcome="impasse (no agreement, each orders separately)" if "impasse" in self.possible_outcomes[0].lower() else self._find_impasse_outcome(),
            )

        # 2. Controllo Intento tramite Zero-Shot NLI
        if latest_features:
            intent = latest_features.get("intent_label", "ongoing")
            
            # Se l'agente dice chiaramente di rinunciare
            if intent == "impasse":
                logger.info("Scenario '%s': NLI impasse intent detected.", self.id)
                return TerminationResult(
                    terminated=True,
                    reason="impasse_signal",
                    outcome=self._find_impasse_outcome(),
                )
            
            # Se l'agente esprime la volontà di chiudere un accordo
            if intent == "agreement":
                agreed_outcome = self._detect_agreement(history)
                if agreed_outcome:
                    names = [t.speaker_name for t in history[-2:]]
                    logger.info("Scenario '%s': NLI agreement on '%s' between %s.", self.id, agreed_outcome, names)
                    return TerminationResult(
                        terminated=True,
                        reason="agreement",
                        outcome=agreed_outcome,
                        agreed_by=names,
                    )

        return TerminationResult(terminated=False, reason="ongoing")

    def _detect_agreement(self, history: list["Turn"]) -> str | None:
        if len(history) < 2: return None
        recent = history[-4:] if len(history) >= 4 else history
        a_turns = [t for t in recent if t.speaker == "agent_a"]
        b_turns = [t for t in recent if t.speaker == "agent_b"]
        
        if not a_turns or not b_turns: return None
        a_text = " ".join(t.utterance.lower() for t in a_turns)
        b_text = " ".join(t.utterance.lower() for t in b_turns)
        
        for outcome in self.possible_outcomes:
            if outcome.lower().startswith("impasse"): continue
            keywords = self._outcome_keywords(outcome)
            if any(kw in a_text for kw in keywords) and any(kw in b_text for kw in keywords):
                return outcome
        return None

    def _outcome_keywords(self, outcome: str) -> list[str]:
        clean = re.sub(r"\(.*?\)", "", outcome).strip().lower()
        words = [w for w in clean.split() if len(w) > 3]
        return [clean] + words

    def _find_impasse_outcome(self) -> str:
        for outcome in self.possible_outcomes:
            if "impasse" in outcome.lower():
                return outcome
        return "impasse"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "game_theory_type": self.game_theory_type,
            "possible_outcomes": self.possible_outcomes,
            "max_rounds": self.max_rounds,
            "agent_a": self._data["agent_a"]["name"],
            "agent_b": self._data["agent_b"]["name"],
        }
    
class ScenarioLoader:
    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        self._scenarios: dict[str, Scenario] = {}
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Scenarios file not found: {self.json_path}")
        
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            
        if not isinstance(raw, list):
            raise ValueError("scenarios.json must be a JSON array of scenario objects.")
            
        for entry in raw:
            try:
                scenario = Scenario(entry)
                self._scenarios[scenario.id] = scenario
                logger.debug("Loaded scenario: %s", scenario.id)
            except ValueError as e:
                logger.error("Skipping invalid scenario: %s", e)
                
        logger.info("Loaded %d scenarios from %s.", len(self._scenarios), self.json_path)

    def get(self, scenario_id: str) -> Scenario:
        if scenario_id not in self._scenarios:
            available = list(self._scenarios.keys())
            raise KeyError(
                f"Scenario '{scenario_id}' not found. Available: {available}"
            )
        return self._scenarios[scenario_id]

    def get_all(self) -> list[Scenario]:
        return list(self._scenarios.values())

    def list_ids(self) -> list[str]:
        return list(self._scenarios.keys())

    def __len__(self) -> int:
        return len(self._scenarios)

    def __repr__(self) -> str:
        return f"ScenarioLoader(path='{self.json_path}', loaded={len(self)})"