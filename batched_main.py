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

for scenario_id in loader.list_ids():
    scenario = loader.get(scenario_id)

    agent = DualPersonaAgent(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        agent_a_config=scenario.get_agent_config("agent_a"),
        agent_b_config=scenario.get_agent_config("agent_b"),
    )

    runner = SimulationRunner(agent, nlp_layer, prompt_builder, judge)

    results = runner.run_batch(
        scenario=scenario,
        n_runs=10,
        run_id_prefix=scenario_id,
        verbose=False,
    )

    for result in results:
        logger.log(result)