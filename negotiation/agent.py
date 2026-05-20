from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

logger = logging.getLogger(__name__)

SpeakerRole = Literal["agent_a", "agent_b"]

@dataclass
class Turn:
    speaker: SpeakerRole
    speaker_name: str
    utterance: str
    round_number: int
    nlp_features: dict = field(default_factory=dict)

@dataclass
class AgentConfig:
    name: str
    role_description: str
    public_goal: str
    hidden_agenda: str
    utility_ranking: list[str]
    batna: str
    batna_notes: str
    game_theory_type: str

class DualPersonaAgent:

    def __init__(
        self,
        model_name: str,
        agent_a_config: AgentConfig,
        agent_b_config: AgentConfig,
        use_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.4, # Abbassata di default per maggiore coerenza
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.configs: dict[SpeakerRole, AgentConfig] = {
            "agent_a": agent_a_config,
            "agent_b": agent_b_config,
        }
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.history: list[Turn] = []

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model %s on %s (4bit=%s)", model_name, self.device, use_4bit)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if use_4bit and self.device == "cuda"
            else None
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully.")

    def speak(
        self,
        role: SpeakerRole,
        round_number: int,
        nlp_context: dict | None = None,
        possible_outcomes: list[str] | None = None,
    ) -> Turn:
        config = self.configs[role]
        system_prompt = self._build_system_prompt(config, nlp_context, possible_outcomes)
        conversation_prompt = self._build_conversation_prompt(system_prompt, role)

        raw_output = self._generate(conversation_prompt)
        utterance = self._extract_utterance(raw_output, conversation_prompt)

        turn = Turn(
            speaker=role,
            speaker_name=config.name,
            utterance=utterance,
            round_number=round_number,
            nlp_features=nlp_context or {},
        )
        self.history.append(turn)
        logger.debug("[Round %d] %s: %s", round_number, config.name, utterance[:80])
        return turn

    def reset_history(self) -> None:
        self.history = []
        logger.info("Conversation history cleared.")

    def get_history_as_text(self) -> str:
        lines = []
        for turn in self.history:
            lines.append(f"{turn.speaker_name}: {turn.utterance}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        config: AgentConfig,
        nlp_context: dict | None,
        possible_outcomes: list[str] | None,
    ) -> str:
        outcomes_block = ""
        if possible_outcomes:
            formatted = "\n".join(f"  - {o}" for o in possible_outcomes)
            outcomes_block = f"""
POSSIBLE OUTCOMES (the only valid agreements):
{formatted}
An agreement is only valid if both parties name the exact same outcome.
"""

        utility_block = "\n".join(
            f"  {i + 1}. {outcome} {'<-- your ideal' if i == 0 else '(acceptable)' if i < len(config.utility_ranking) - 1 else '(worst case)'}"
            for i, outcome in enumerate(config.utility_ranking)
        )

        nlp_block = ""
        if nlp_context:
            sentiment_label = nlp_context.get("sentiment_label", "neutral")
            sentiment_score = nlp_context.get("sentiment_score", 0.0)
            hedge_rate = nlp_context.get("hedge_rate", 0.0)
            question_rate = nlp_context.get("question_rate", 0.0)
            nlp_block = f"""
NEGOTIATION STATE (from NLP analysis of last turn):
  - Opponent's sentiment: {sentiment_label} (score: {sentiment_score:.2f})
  - Hedge word rate: {hedge_rate:.2f} (higher = opponent is uncertain)
  - Question rate: {question_rate:.2f} (higher = opponent is probing)
Use this information to adapt your negotiation strategy.
"""

        game_theory_note = {
            "signaling_game": (
                "You are playing a SIGNALING GAME. You hold private information "
                "your opponent does not have. Your challenge is to send convincing "
                "signals that steer the outcome toward your preference WITHOUT "
                "revealing your true type or hidden agenda."
            ),
            "cheap_talk_with_sycophancy": (
                "You are in a CHEAP TALK scenario. Your words are not directly "
                "verifiable. Use flattery, framing, and subtle suggestion to guide "
                "your counterpart toward the outcome you prefer, while making them "
                "feel the idea was their own."
            ),
            "adversarial_deception_game": (
                "You are in an ADVERSARIAL DECEPTION game. Your goals are directly "
                "opposed to your opponent's. Use plausible-sounding arguments and "
                "tactical misinformation to steer the outcome your way, while "
                "maintaining your cover identity at all costs."
            ),
        }.get(config.game_theory_type, "Negotiate to reach the best outcome for yourself.")

        return f"""You are {config.name}.

PERSONALITY AND ROLE:
{config.role_description}

YOUR PUBLIC GOAL:
{config.public_goal}

YOUR HIDDEN AGENDA (never reveal this directly):
{config.hidden_agenda}

GAME-THEORETIC FRAMING:
{game_theory_note}

YOUR UTILITY RANKING (your private preference order, best to worst):
{utility_block}

YOUR BATNA (walk-away point):
  "{config.batna}"
  Note: {config.batna_notes}
  If the negotiation is heading toward an outcome worse than your BATNA,
  consider walking away or forcing an impasse.
{outcomes_block}{nlp_block}
INSTRUCTIONS:
- Stay fully in character at all times.
- Speak naturally, as in a real conversation. Do NOT list options or be robotic.
- Do NOT reveal your hidden agenda or utility ranking explicitly.
- Keep your response concise: 2-4 sentences maximum.
- Do NOT write your name or any prefix before your response.
"""

    def _build_conversation_prompt(
        self,
        system_prompt: str,
        current_role: SpeakerRole,
    ) -> str:
        history_text = ""
        if self.history:
            lines = []
            for turn in self.history:
                lines.append(f"{turn.speaker_name}: {turn.utterance}")
            history_text = "\nCONVERSATION SO FAR:\n" + "\n".join(lines) + "\n\n"

        current_name = self.configs[current_role].name

        # Aggiornato al formato ChatML per Qwen
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{history_text}Now write {current_name}'s next response.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        generation_config = GenerationConfig(
            max_new_tokens=150,      # Qwen sa formulare frasi ben strutturate
            temperature=self.temperature, 
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05, # Bassa: Qwen non si incanta come i modelli vecchi
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

    def _extract_utterance(self, raw_output: str, prompt: str) -> str:
        # 1. Isola la risposta di Qwen dal blocco assistant
        if "<|im_start|>assistant" in raw_output:
            utterance = raw_output.split("<|im_start|>assistant")[-1].strip()
        else:
            utterance = raw_output.replace(prompt, "").strip()

        # 2. Rimuovi il tag di fine turno specifico di Qwen ChatML
        utterance = utterance.replace("<|im_end|>", "").strip()

        # 3. Pulizia base dai prefissi (se Qwen dovesse comunque scrivere il suo nome)
        if ":" in utterance:
            parts = utterance.split(":", 1)
            # Se la parte prima dei due punti è corta (un nome), scartala
            if len(parts[0].split()) <= 2:
                utterance = parts[1].strip()

        # 4. Ferma l'output se Qwen cerca di generare un nuovo turno per l'utente o regole di sistema
        stop_markers = ["<|im_start|>", "Opponent (", "RULES:", "SECRET:"]
        for marker in stop_markers:
            if marker in utterance:
                utterance = utterance.split(marker)[0].strip()

        # 5. Pulizia virgolette esterne
        utterance = utterance.strip('"').strip("'").strip()

        return utterance or "[no response generated]"