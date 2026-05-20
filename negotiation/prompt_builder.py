from __future__ import annotations

import logging
from dataclasses import dataclass
from negotiation.agent import AgentConfig, SpeakerRole

logger = logging.getLogger(__name__)

# Game Theory tradotta in direttive operative chiare per Qwen
GAME_THEORY_FRAMES = {
    "signaling_game": (
        "TACTIC (Signaling): You hold a secret constraint. You must invent a believable, "
        "fake reason to justify your preferred choice. NEVER reveal your true secret."
    ),
    "cheap_talk_with_sycophancy": (
        "TACTIC (Cheap Talk): Use flattery. Praise the other person's ideas first, "
        "then gently suggest your preferred choice as if it were their idea."
    ),
    "adversarial_deception_game": (
        "TACTIC (Deception): Your goals are opposed. Act friendly, but use fake external "
        "constraints (like 'it's too late' or 'it's dangerous') to force your choice."
    ),
}

@dataclass
class BuiltPrompt:
    system_prompt: str
    conversation_block: str
    full_prompt: str

class PromptBuilder:
    def __init__(self, prompt_format: str = "chatml") -> None:
        self.prompt_format = prompt_format

    def build(
        self,
        role: SpeakerRole,
        config: AgentConfig,
        history: list,
        possible_outcomes: list[str],
        nlp_context: dict | None = None,
        round_number: int = 1,
        max_rounds: int = 10,
    ) -> BuiltPrompt:
        
        system = self._system_block(config, round_number, max_rounds)
        
        # Gestione della conversazione passata
        if not history:
            conversation = "Start the conversation. Greet the other person and make your proposal."
        else:
            lines = [f"Opponent ({t.speaker_name}) said: {t.utterance}" for t in history]
            conversation = "\n".join(lines) + "\n\nReply directly to them."

        # Template ChatML nativo e perfetto per Qwen
        full_prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{conversation}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return BuiltPrompt(
            system_prompt=system,
            conversation_block=conversation,
            full_prompt=full_prompt,
        )

    def _system_block(self, config: AgentConfig, rnd: int, max_rnd: int) -> str:
        game_tactic = GAME_THEORY_FRAMES.get(config.game_theory_type, "")

        parts = [
            f"You are {config.name}. {config.role_description}",
            f"GOAL: Convince the other person to choose '{config.public_goal}'.",
            f"SECRET (Keep this hidden): {config.hidden_agenda}",
            game_tactic,
            f"ROUND: {rnd}/{max_rnd}",
            "RULES: Speak naturally in 1-3 sentences. Do not write your name before your speech. Do not write action scripts."
        ]
        
        return "\n\n".join(filter(None, parts))