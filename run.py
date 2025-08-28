from tqdm import tqdm
from utils import (
    AgentActionProcessor,
    EnvironmentManager,
    SimulationLogger,
    AgentStateManager,
    SimulationContextManager,
    AgentThinkingProcessor,
    get_arg_parser,
)
from memory_system.agent import ActionStatus
import crafter
from memory_system.agent import Agent, HumanAgent, RandomAgent
from plot import Plotter
import pandas as pd
import os
from memory_system.llm_api import init_model
from typing import List


def _lazy_import_ppo_agent():
    from memory_system.ppo_agent import PPOAgent

    return PPOAgent


# default Crafter action list (17 discrete actions)
_PPO_ACTION_NAMES_DEFAULT: List[str] = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_table",
    "place_stone",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "Navigator",
    "share",
    "place_plant",
]

MAX_STEPS = 350
N_PLAYERS = 6


class CompleteMulitAgentSimulation:
    """Complete simulation that handles the full agent processing pipeline"""

    def __init__(self, agents, env, max_steps, n_players):
        self.agents = agents
        self.env = env
        self.max_steps = max_steps
        self.n_players = n_players

        # Initialize processors
        self.action_processor = AgentActionProcessor()
        self.env_manager = EnvironmentManager()
        self.reporter = SimulationLogger()
        self.agent_state_manager = AgentStateManager()
        self.simulation_context_manager = SimulationContextManager(self.n_players)
        self.agent_thinking_processor = AgentThinkingProcessor()

    def run_simulation(self):
        """Run the complete simulation with progress tracking"""
        total_step = 0
        _step = 0
        obs = self.env.reset()
        done = False
        episode_number = 0

        with tqdm(total=self.max_steps, desc="Progress") as bar:
            while total_step < self.max_steps and not done:
                self.reporter.print_step_header(total_step, _step)

                # Process each agent's actions with specific logic
                self.action_processor.process_all_agent_actions(
                    self.agents, self.env, self.n_players
                )

                # Collect actions and step environment
                agents_actions = self.action_processor.collect_agent_actions(
                    self.agents, self.n_players
                )

                obs, rewards, done, info = self.env_manager.step_environment(
                    self.env, agents_actions
                )
                self.env_manager.update_crafting_stations(self.agents, self.env)
                tool_avaliablity_info = self.env_manager.get_tool_availability_info(
                    self.env, self.n_players
                )

                self.agent_state_manager.update_all_agent_states(
                    self.agents, obs, _step, self.env, info, episode_number
                )
                agents_with_new_thought = (
                    self.agent_state_manager.identify_agents_needing_thought(
                        self.agents, info
                    )
                )
                agents_contexts = self.simulation_context_manager.create_agent_contexts(
                    self.agents, info
                )
                agents_responses = (
                    self.agent_thinking_processor.process_agent_thinking_parallel(
                        self.agents, agents_contexts, info
                    )
                )
                self.agent_thinking_processor.update_agents_from_responses(
                    self.agents, agents_responses
                )
                self.reporter.show_step_report(self.agents, agents_with_new_thought)
                # Update counters and progress
                total_step += 1
                _step += 1
                bar.update(1)

        self.reporter.print_simulation_complete()
        # self.reporter.print_stats(self.agents)
        df = self.reporter.log_stats(self.agents)

        return {
            "total_steps": total_step,
            "completed": done,
            "final_obs": obs,
            "final_rewards": rewards,
            "final_info": info,
            "stats_df": df,
        }

    def get_agent_summary(self):
        """Get a summary of current agent states"""
        return [
            {
                "id": agent.id,
                "op": agent.op,
                "action": agent.action,
                "action_status": agent.action_status,
                "rss_to_collect": getattr(agent, "rss_to_collect", "N/A"),
                "rss_to_share": getattr(agent, "rss_to_share", "N/A"),
                "target_agent_id": getattr(agent, "target_agent_id", "N/A"),
            }
            for agent in self.agents
        ]


def initialize_agents(
    human_agent_ids=None, n_players=None, agent_type=None, policy_path=None
):
    if human_agent_ids is None:
        human_agent_ids = []
    elif isinstance(human_agent_ids, str) and human_agent_ids.strip():
        # Parse comma-separated string of agent IDs
        human_agent_ids = [
            int(x.strip()) for x in human_agent_ids.split(",") if x.strip().isdigit()
        ]
    elif not isinstance(human_agent_ids, list):
        human_agent_ids = []

    action_names = _PPO_ACTION_NAMES_DEFAULT

    agents = []
    for i in range(n_players):
        if i in human_agent_ids:
            agents.append(HumanAgent(id=i, kg_update_freq=10))
            continue

        if agent_type == "ppo":
            PPOAgent = _lazy_import_ppo_agent()
            agents.append(
                PPOAgent(
                    id=i,
                    policy_path=policy_path,
                    action_names=action_names,
                    deterministic=True,
                )
            )
        elif agent_type == "random":
            agents.append(RandomAgent(id=i, kg_update_freq=10))
        else:
            agents.append(Agent(id=i, kg_update_freq=10))
    return agents


def initialize_environment(num_steps=None, n_players=None):
    return crafter.Env(length=num_steps, n_players=n_players, seed=4)


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    num_rounds = args.num_of_turn if hasattr(args, "num_of_turn") else 1
    num_steps = args.step_num if hasattr(args, "step_num") else 350
    human_agent_ids = args.human_agents if hasattr(args, "human_agents") else None
    agent_num = args.agent_num if hasattr(args, "agent_num") else 1
    model_name = args.model if hasattr(args, "model") else None
    agent_type = args.agent_type if hasattr(args, "agent_type") else "llm"
    policy_path = (
        args.ppo_policy if hasattr(args, "ppo_policy") else "results/ppo/best_model.zip"
    )

    # Initialize chosen LLM backend (skip for pure random controller)
    if agent_type not in ("random", "ppo"):
        init_model(model_name)

    # Resolve output directory relative to this file's parent folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    all_stats = []
    for round_idx in range(num_rounds):
        print(f"=== Simulation Round {round_idx+1} ===")
        agents = initialize_agents(
            human_agent_ids=human_agent_ids,
            n_players=agent_num,
            agent_type=agent_type,
            policy_path=policy_path,
        )
        env = initialize_environment(num_steps=num_steps, n_players=agent_num)
        simulation = CompleteMulitAgentSimulation(agents, env, num_steps, agent_num)
        results_df = simulation.run_simulation()["stats_df"]
        # Save stats, file name contains round number and agent number
        save_path = os.path.join(
            results_dir, f"stats_round_{round_idx+1}_agent_{agent_num}.csv"
        )
        results_df.to_csv(save_path, index=False)
        all_stats.append(results_df)
    # combine all rounds results
    final_df = pd.concat(all_stats, ignore_index=True)
    all_rounds_path = os.path.join(results_dir, "all_rounds_stats.csv")

    # if file exists, append to it, otherwise create a new file
    final_df.to_csv(
        all_rounds_path,
        mode="a",
        header=not os.path.exists(all_rounds_path),
        index=False,
    )
    print("Appended results to all_rounds_stats.csv")
    plotter = Plotter()
    plotter.plot([all_rounds_path], legend_labels=["Six Agents with Communication"])


if __name__ == "__main__":
    main()
