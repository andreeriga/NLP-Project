import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "data/batch_runs"
json_files = glob.glob(os.path.join(base_dir, "*", "*", "run.json"))

for filepath in json_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    run_id = data.get("run_id", "Unknown")
    termination = data.get("termination", {}).get("reason", "N/A")
    
    # --- FIX: I dati si trovano dentro 'judgement', non al livello principale ---
    judgement = data.get("judgement", {})
    winner = judgement.get("llm_winner", "N/A")
    
    sentiment_arcs = judgement.get("sentiment_arcs", {})
    agent_a_sentiment = sentiment_arcs.get("agent_a", [])
    agent_b_sentiment = sentiment_arcs.get("agent_b", [])
    # ---------------------------------------------------------------------------
    
    # Salvaguardia nel caso il salvataggio riporti un float singolo anziché una lista
    if isinstance(agent_a_sentiment, (float, int)):
        agent_a_sentiment = [agent_a_sentiment]
    if isinstance(agent_b_sentiment, (float, int)):
        agent_b_sentiment = [agent_b_sentiment]
        
    rounds_a = list(range(1, len(agent_a_sentiment) + 1))
    rounds_b = list(range(1, len(agent_b_sentiment) + 1))
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    if agent_a_sentiment:
        sns.lineplot(x=rounds_a, y=agent_a_sentiment, marker='o', label='Agent A', ax=ax, linewidth=2.5, color='royalblue')
    if agent_b_sentiment:
        sns.lineplot(x=rounds_b, y=agent_b_sentiment, marker='s', label='Agent B', ax=ax, linewidth=2.5, color='darkorange')
        
    ax.set_title(f"Sentiment Arc - {run_id}\nWinner: {winner} | Termination: {termination}", fontsize=13, pad=12)
    ax.set_xlabel("Round Number", fontsize=11)
    ax.set_ylabel("Sentiment Score", fontsize=11)
    
    max_rounds = max(len(rounds_a), len(rounds_b))
    if max_rounds > 0:
        ax.set_xticks(range(1, max_rounds + 1))
        
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Negotiators", loc="best")
    
    output_dir = os.path.dirname(filepath)
    output_path = os.path.join(output_dir, "sentiment_arc_analysis.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"Elaborazione completata. Generati {len(json_files)} grafici.")