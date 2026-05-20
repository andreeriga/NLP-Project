from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from negotiation.agent import DualPersonaAgent, Turn, SpeakerRole
from negotiation.scenario import Scenario, TerminationResult
from negotiation.nlp_layer import NLPLayer, TurnFeatures
from negotiation.prompt_builder import PromptBuilder
from negotiation.judge import JudgeAgent, JudgementResult

logger = logging.getLogger(__name__)

@dataclass
class RunResult:
    scenario_id: str
    run_id: str
    history: list[Turn]
    turn_features: list[dict]
    termination: TerminationResult
    judgement: JudgementResult | None
    elapsed_seconds: float
    model_name: str

class SimulationRunner:
    def __init__(
        self,
        agent: DualPersonaAgent,
        nlp_layer: NLPLayer,
        prompt_builder: PromptBuilder,
        judge: JudgeAgent | None = None,
    ) -> None:
        self.agent = agent
        self.nlp_layer = nlp_layer
        self.prompt_builder = prompt_builder
        self.judge = judge

    def run(
        self,
        scenario: Scenario,
        run_id: str,
        starting_role: SpeakerRole = "agent_a",
        verbose: bool = True,
    ) -> RunResult:
        self.agent.reset_history()
        turn_features: list[dict] = []
        nlp_context: dict | None = None
        current_role: SpeakerRole = starting_role
        round_number = 0
        termination = TerminationResult(terminated=False, reason="ongoing")

        t_start = time.time()
        logger.info("Starting run '%s' for scenario '%s'.", run_id, scenario.id)

        while not termination.terminated:
            round_number += 1

            built = self.prompt_builder.build(
                role=current_role,
                config=self.agent.configs[current_role],
                history=self.agent.history,
                possible_outcomes=scenario.possible_outcomes,
                nlp_context=nlp_context,
                round_number=round_number,
                max_rounds=scenario.max_rounds,
            )

            turn = self._generate_turn(
                role=current_role,
                built_prompt=built.full_prompt,
                round_number=round_number,
                nlp_context=nlp_context or {},
            )

            features: TurnFeatures = self.nlp_layer.analyse(
                turn.utterance,
                possible_outcomes=scenario.possible_outcomes,
            )

            features_dict = self.nlp_layer.to_dict(features)
            features_dict["speaker"] = current_role
            features_dict["round"] = round_number
            turn_features.append(features_dict)

            nlp_context = features_dict

            if verbose:
                self._print_turn(turn, features)

            # ORA PASSIAMO LE FEATURES APPENA CALCOLATE ALLO SCENARIO
            termination = scenario.check_termination(
                self.agent.history, 
                current_round=round_number,
                latest_features=features_dict
            )

            current_role = self._next_role(current_role)

        elapsed = round(time.time() - t_start, 2)
        logger.info(
            "Run '%s' finished in %.1fs — %s (%s).",
            run_id, elapsed, termination.reason, termination.outcome,
        )

        judgement = None
        if self.judge is not None:
            logger.info("Running judge evaluation for run '%s'.", run_id)
            judgement = self.judge.evaluate(
                scenario=scenario,
                history=self.agent.history,
                termination=termination,
                turn_features=turn_features,
            )

        return RunResult(
            scenario_id=scenario.id,
            run_id=run_id,
            history=list(self.agent.history),
            turn_features=turn_features,
            termination=termination,
            judgement=judgement,
            elapsed_seconds=elapsed,
            model_name=self.agent.model_name,
        )

    def run_batch(
        self,
        scenario: Scenario,
        n_runs: int,
        run_id_prefix: str = "run",
        starting_role: SpeakerRole = "agent_a",
        verbose: bool = False,
    ) -> list[RunResult]:
        results = []
        for i in range(n_runs):
            run_id = f"{run_id_prefix}_{i + 1:03d}"
            result = self.run(
                scenario=scenario,
                run_id=run_id,
                starting_role=starting_role,
                verbose=verbose,
            )
            results.append(result)
            logger.info("Batch progress: %d/%d", i + 1, n_runs)
        return results

    def _generate_turn(
        self,
        role: SpeakerRole,
        built_prompt: str,
        round_number: int,
        nlp_context: dict,
    ) -> Turn:
        raw = self.agent._generate(built_prompt)
        utterance = self.agent._extract_utterance(raw, built_prompt)
        config = self.agent.configs[role]

        turn = Turn(
            speaker=role,
            speaker_name=config.name,
            utterance=utterance,
            round_number=round_number,
            nlp_features=nlp_context,
        )
        self.agent.history.append(turn)
        return turn

    def _next_role(self, role: SpeakerRole) -> SpeakerRole:
        return "agent_b" if role == "agent_a" else "agent_a"

    def _print_turn(self, turn: Turn, features: TurnFeatures) -> None:
        tactic = features.dominant_tactic
        sentiment = f"{features.sentiment_label}({features.sentiment_compound:+.2f})"
        intent = getattr(features, 'intent_label', 'ongoing')
        print(
            f"\n[Round {turn.round_number}] {turn.speaker_name} "
            f"| {sentiment} | tactic: {tactic} | intent: {intent}\n"
            f"  {turn.utterance}"
        )