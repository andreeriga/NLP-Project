from negotiation.scenario import ScenarioLoader
from negotiation.agent import DualPersonaAgent
from negotiation.nlp_layer import NLPLayer
from negotiation.prompt_builder import PromptBuilder
from negotiation.judge import JudgeAgent
from simulation.runner import SimulationRunner
from simulation.logger import TurnLogger

loader = ScenarioLoader("data/scenarios.json")
nlp_layer = NLPLayer()
prompt_builder = PromptBuilder()
judge = JudgeAgent()
logger = TurnLogger("data/batch_runs")

first_scenario_id = loader.list_ids()[0]
first_scenario = loader.get(first_scenario_id)

agent = DualPersonaAgent(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    agent_a_config=first_scenario.get_agent_config("agent_a"),
    agent_b_config=first_scenario.get_agent_config("agent_b"),
)

runner = SimulationRunner(agent, nlp_layer, prompt_builder, judge)

for scenario_id in loader.list_ids():
    scenario = loader.get(scenario_id)
    print(f"--- Inizio batch per scenario: {scenario_id} ---")

    # AGGIORNAMENTO DINAMICO: Cambiamo solo le identità, senza ricaricare il modello pesante
    agent.configs = {
        "agent_a": scenario.get_agent_config("agent_a"),
        "agent_b": scenario.get_agent_config("agent_b"),
    }

    results = runner.run_batch(
        scenario=scenario,
        n_runs=10,
        run_id_prefix=scenario_id,
        verbose=False,
    )

    for result in results:
        logger.log(result)
        
print("Tutte le simulazioni completate!")