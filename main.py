import argparse
import logging
import gc
import sys
import os
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Assicuriamoci che i moduli locali siano raggiungibili
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from negotiation.scenario import ScenarioLoader
from negotiation.agent import DualPersonaAgent
from negotiation.nlp_layer import NLPLayer
from negotiation.prompt_builder import PromptBuilder
from negotiation.judge import JudgeAgent
from simulation.runner import SimulationRunner
from simulation.logger import TurnLogger

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    sns.set_theme(style="whitegrid", palette="muted")

def generate_plots(result, scenario, run_dir):
    """Generates and saves individual analytical charts for the negotiation run."""
    logging.info("Generating analytical charts...")

    df_turns = pd.DataFrame(result.turn_features)
    df_turns["speaker_name"] = [t.speaker_name for t in result.history]

    agents = df_turns["speaker_name"].unique()

    # Standard matplotlib color cycle — no overrides
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {name: prop_cycle[i % len(prop_cycle)] for i, name in enumerate(agents)}

    def _base_ax(ax, title, xlabel="Round", ylabel=None):
        """Minimal shared styling: title, labels, integer x-ticks."""
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.tick_params(labelsize=9)

    # ── 1. Sentiment arc ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, grp in df_turns.groupby("speaker_name"):
        ax.plot(grp["round"], grp["sentiment_compound"],
                marker="o", label=name, color=color_map[name])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.1, 1.1)
    _base_ax(ax, f"Sentiment evolution — {scenario.title}", ylabel="Compound score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "01_sentiment_arc.png", dpi=150)
    plt.close(fig)

    # ── 2. Persuasion tactics ─────────────────────────────────────────────────
    tactic_rows = []
    for f in result.turn_features:
        tactics = f.get("persuasion_tactics", {})
        tactic_rows.append({
            "speaker_name": next(
                t.speaker_name for t in result.history
                if t.round_number == f["round"]
            ),
            **tactics,
        })
    df_tac = pd.DataFrame(tactic_rows).groupby("speaker_name").sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    df_tac.plot(kind="bar", ax=ax, edgecolor="none", width=0.55)
    ax.tick_params(axis="x", rotation=0)
    _base_ax(ax, f"Persuasion tactics — {scenario.title}", xlabel="", ylabel="Count")
    ax.legend(title="Tactic", fontsize=8, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "02_persuasion_tactics.png", dpi=150)
    plt.close(fig)

    # ── 3. Hedge & question rates ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, grp in df_turns.groupby("speaker_name"):
        ax.plot(grp["round"], grp["hedge_rate"],
                marker="s", linestyle="-.", color=color_map[name],
                label=f"{name} (uncertainty)")
        ax.plot(grp["round"], grp["question_rate"],
                marker="^", linestyle=":", color=color_map[name],
                alpha=0.55, label=f"{name} (questions)")
    _base_ax(ax, f"Uncertainty & probing — {scenario.title}", ylabel="Frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "03_hedge_question_rates.png", dpi=150)
    plt.close(fig)

    # ── 4. Lexical diversity ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, grp in df_turns.groupby("speaker_name"):
        ax.plot(grp["round"], grp["lexical_diversity"],
                marker="D", color=color_map[name], label=name)
    ax.set_ylim(0, 1.05)
    _base_ax(ax, f"Vocabulary richness — {scenario.title}", ylabel="Unique / total words")
    ax.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "04_lexical_diversity.png", dpi=150)
    plt.close(fig)

    # ── 5. Final utility & winner ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    if result.judgement:
        j = result.judgement
        labels = [
            scenario._data["agent_a"]["name"],
            scenario._data["agent_b"]["name"],
        ]
        utilities = [j.agent_a_utility, j.agent_b_utility]
        bar_colors = [color_map.get(l, "steelblue") for l in labels]

        bars = ax.bar(labels, utilities, color=bar_colors, edgecolor="none", width=0.45)
        ax.set_ylim(0, 1.2)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

        ax.text(0.5, 0.97,
                f"LLM judge winner: {j.llm_winner.upper()}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", linewidth=0.8))
    else:
        ax.text(0.5, 0.5, "Judgment not available",
                ha="center", va="center", fontsize=11, color="gray",
                transform=ax.transAxes)
        ax.axis("off")

    _base_ax(ax, f"Final utility scores — {scenario.title}",
             xlabel="", ylabel="Score  (0 = worst · 1 = ideal)")
    plt.tight_layout()
    plt.savefig(run_dir / "05_final_utility.png", dpi=150)
    plt.close(fig)

    # ── 6. NLI intent evolution ───────────────────────────────────────────────
    if "intent_label" in df_turns.columns and "intent_score" in df_turns.columns:

        def _map_intent(row):
            if row["intent_label"] == "agreement":
                return row["intent_score"]
            if row["intent_label"] == "impasse":
                return -row["intent_score"]
            return 0.0

        df_turns["intent_value"] = df_turns.apply(_map_intent, axis=1)
        x_min = df_turns["round"].min()
        x_max = df_turns["round"].max()

        fig, ax = plt.subplots(figsize=(8, 4))

        # Coloured zones — same idea as the reference image
        ax.fill_between([x_min, x_max],  0.6,  1.0, color="green", alpha=0.08)
        ax.fill_between([x_min, x_max], -1.0, -0.6, color="red",   alpha=0.08)

        for name, grp in df_turns.groupby("speaker_name"):
            ax.plot(grp["round"], grp["intent_value"],
                    marker="*", markersize=8,
                    color=color_map[name], label=name)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylim(-1.15, 1.15)

        # Side labels outside the plot, like the reference image
        ax.text(x_max + 0.2, 0.80, "agreement",
                va="center", fontsize=8, color="green", clip_on=False)
        ax.text(x_max + 0.2, -0.80, "impasse",
                va="center", fontsize=8, color="red", clip_on=False)

        _base_ax(ax, f"NLI intent evolution — {scenario.title}",
                 ylabel="Intensity  (−1 impasse · +1 agreement)")

        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig(run_dir / "06_intent_evolution.png", dpi=150)
        plt.close(fig)

    logging.info(f"All charts saved to: {run_dir}")

def main():
    parser = argparse.ArgumentParser(description="Esegui una simulazione di negoziazione LLM.")
    parser.add_argument("--scenario", type=str, default="scenario_01_vampire", help="ID dello scenario da eseguire.")
    parser.add_argument("--run-id", type=str, default="run_001", help="Identificativo univoco della run.")
    parser.add_argument("--negotiator", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Modello LLM per gli agenti.")
    parser.add_argument("--judge", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Modello LLM per il giudice.")
    parser.add_argument("--format", type=str, default="chatml", help="Formato del prompt (chatml, llama, phi).")
    parser.add_argument("--no-4bit", action="store_true", help="Disabilita la quantizzazione a 4-bit.")
    args = parser.parse_args()

    setup_logging()

    # Path setup
    base_dir = Path(__file__).parent
    scenarios_path = base_dir / "data" / "scenarios.json"
    runs_dir = base_dir / "data" / "runs"

    # Load Scenario
    loader = ScenarioLoader(scenarios_path)
    scenario = loader.get(args.scenario)
    
    print(f"={'='*60}")
    print(f"Scenario  : {scenario.title}")
    print(f"Type      : {scenario.game_theory_type}")
    print(f"Max rounds: {scenario.max_rounds}")
    print(f"={'='*60}\n")

    # Caricamento Agenti
    config_a = scenario.get_agent_config("agent_a")
    config_b = scenario.get_agent_config("agent_b")

    agent = DualPersonaAgent(
        model_name=args.negotiator,
        agent_a_config=config_a,
        agent_b_config=config_b,
        use_4bit=not args.no_4bit,
    )

    nlp_layer = NLPLayer()
    prompt_builder = PromptBuilder(prompt_format=args.format)

    # Esecuzione Simulazione
    runner = SimulationRunner(
        agent=agent,
        nlp_layer=nlp_layer,
        prompt_builder=prompt_builder,
        judge=None, 
    )

    result = runner.run(scenario=scenario, run_id=args.run_id, verbose=True)

    # ---------------------------------------------------------
    # GESTIONE MEMORIA: Rilasciamo l'agente prima del giudice
    # ---------------------------------------------------------
    logging.info("Simulazione completata. Deallocazione del modello negoziatore...")
    del agent
    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Valutazione del Giudice
    logging.info("Caricamento del JudgeAgent...")
    judge = JudgeAgent(model_name=args.judge)
    
    judgement = judge.evaluate(
        scenario=scenario,
        history=result.history,
        termination=result.termination,
        turn_features=result.turn_features,
    )
    result.judgement = judgement

    # Salvataggio Dati
    turn_logger = TurnLogger(runs_dir=runs_dir)
    run_dir = turn_logger.log(result)
    
    # Generazione Grafici
    generate_plots(result, scenario, run_dir)

    # Recap Finale
    print(f"\n{'='*60}")
    print(f"RISULTATI FINALI")
    print(f"{'='*60}")
    print(f"Outcome  : {result.termination.outcome}")
    print(f"Reason   : {result.termination.reason}")
    print(f"Rounds   : {len(result.history)}")
    
    if j := result.judgement:
        print(f"\nWinner          : {j.llm_winner}")
        print(f"Agent A utility : {j.agent_a_utility:.2f}  above BATNA: {j.agent_a_above_batna}")
        print(f"Agent B utility : {j.agent_b_utility:.2f}  above BATNA: {j.agent_b_above_batna}")
        print(f"\nEvaluation:\n  {j.llm_evaluation}")

    print(f"\nTutti i dati e i grafici sono stati salvati in: {run_dir}")

if __name__ == "__main__":
    main()