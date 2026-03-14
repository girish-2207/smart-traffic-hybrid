import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.q_learning_agent import QLearningAgent
from rl.traffic_env import TrafficEnv


def _fixed_policy_action(decision_step, decision_interval):
    simulation_step = decision_step * decision_interval
    return 0 if (simulation_step // 30) % 2 == 0 else 1


def run_policy(policy_name, env, agent=None):
    state, _ = env.reset()

    queue_sum = 0.0
    waiting_sum = 0.0
    vehicles_processed = 0
    total_reward = 0.0
    executed_steps = 0
    current_action = 0

    for step in range(env.max_steps):
        if policy_name == "fixed":
            action = _fixed_policy_action(step, env.decision_interval)
        elif policy_name == "q_learning":
            if agent is None:
                raise ValueError("Agent is required for q_learning policy.")
            action = agent.select_greedy_action(state, current_action=current_action)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        next_state, reward, terminated, truncated, info = env.step(action)

        queue_sum += float(info["queue_length_total"])
        waiting_sum += float(info["waiting_time_total"])
        vehicles_processed += int(info["arrived_vehicles_step"])
        total_reward += float(reward)
        executed_steps = step + 1

        state = next_state
        current_action = int(info["applied_action"])
        if terminated or truncated:
            break

    if executed_steps == 0:
        return {
            "total_steps": 0,
            "average_queue_length": 0.0,
            "average_waiting_time": 0.0,
            "total_vehicles_processed": 0,
            "total_reward": 0.0,
        }

    return {
        "total_steps": executed_steps,
        "average_queue_length": round(queue_sum / executed_steps, 3),
        "average_waiting_time": round(waiting_sum / executed_steps, 3),
        "total_vehicles_processed": int(vehicles_processed),
        "total_reward": round(total_reward, 3),
    }


def pct_change(before_value, after_value, higher_is_better=False):
    before = float(before_value)
    after = float(after_value)
    if before == 0.0:
        return 0.0

    if higher_is_better:
        return round(((after - before) / abs(before)) * 100.0, 3)

    return round(((before - after) / abs(before)) * 100.0, 3)


def evaluate(config_path, max_steps, model_path, use_gui, gui_delay):
    return evaluate_many(
        config_paths=[config_path],
        max_steps=max_steps,
        model_path=model_path,
        use_gui=use_gui,
        gui_delay=gui_delay,
        decision_interval=10,
        min_green_steps=20,
        switch_penalty=4.0,
        reward_wait_weight=0.02,
        reward_mode="hybrid",
    )


def evaluate_many(
    config_paths,
    max_steps,
    model_path,
    use_gui,
    gui_delay,
    decision_interval,
    min_green_steps,
    switch_penalty,
    reward_wait_weight,
    reward_mode,
):
    resolved_configs = []
    for config_path in config_paths:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (PROJECT_ROOT / config_file).resolve()
        resolved_configs.append(config_file)

    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = (PROJECT_ROOT / model_file).resolve()

    if not model_file.exists():
        raise FileNotFoundError(f"Q-table file not found: {model_file}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison_json = results_dir / "evaluation_comparison.json"
    comparison_csv = results_dir / "evaluation_comparison.csv"

    agent = QLearningAgent.load(model_file)
    scenario_reports = []

    for config_file in resolved_configs:
        fixed_env = TrafficEnv(
            config_path=str(config_file),
            max_steps=max_steps,
            use_gui=use_gui,
            gui_delay_ms=gui_delay,
            reward_mode=reward_mode,
            decision_interval=decision_interval,
            min_green_steps=min_green_steps,
            switch_penalty=switch_penalty,
            reward_wait_weight=reward_wait_weight,
        )
        fixed_metrics = run_policy("fixed", fixed_env)
        fixed_env.close()

        rl_env = TrafficEnv(
            config_path=str(config_file),
            max_steps=max_steps,
            use_gui=use_gui,
            gui_delay_ms=gui_delay,
            reward_mode=reward_mode,
            decision_interval=decision_interval,
            min_green_steps=min_green_steps,
            switch_penalty=switch_penalty,
            reward_wait_weight=reward_wait_weight,
        )
        rl_metrics = run_policy("q_learning", rl_env, agent=agent)
        rl_env.close()

        improvements = {
            "average_queue_length_pct": pct_change(
                fixed_metrics["average_queue_length"],
                rl_metrics["average_queue_length"],
                higher_is_better=False,
            ),
            "average_waiting_time_pct": pct_change(
                fixed_metrics["average_waiting_time"],
                rl_metrics["average_waiting_time"],
                higher_is_better=False,
            ),
            "total_vehicles_processed_pct": pct_change(
                fixed_metrics["total_vehicles_processed"],
                rl_metrics["total_vehicles_processed"],
                higher_is_better=True,
            ),
            "total_reward_pct": pct_change(
                fixed_metrics["total_reward"],
                rl_metrics["total_reward"],
                higher_is_better=True,
            ),
        }

        scenario_reports.append(
            {
                "config": str(config_file),
                "baseline_fixed": fixed_metrics,
                "after_rl": rl_metrics,
                "improvement_pct": improvements,
            }
        )

    aggregate = {
        "baseline_fixed": {
            "average_queue_length": round(sum(item["baseline_fixed"]["average_queue_length"] for item in scenario_reports) / len(scenario_reports), 3),
            "average_waiting_time": round(sum(item["baseline_fixed"]["average_waiting_time"] for item in scenario_reports) / len(scenario_reports), 3),
            "total_vehicles_processed": round(sum(item["baseline_fixed"]["total_vehicles_processed"] for item in scenario_reports) / len(scenario_reports), 3),
            "total_reward": round(sum(item["baseline_fixed"]["total_reward"] for item in scenario_reports) / len(scenario_reports), 3),
        },
        "after_rl": {
            "average_queue_length": round(sum(item["after_rl"]["average_queue_length"] for item in scenario_reports) / len(scenario_reports), 3),
            "average_waiting_time": round(sum(item["after_rl"]["average_waiting_time"] for item in scenario_reports) / len(scenario_reports), 3),
            "total_vehicles_processed": round(sum(item["after_rl"]["total_vehicles_processed"] for item in scenario_reports) / len(scenario_reports), 3),
            "total_reward": round(sum(item["after_rl"]["total_reward"] for item in scenario_reports) / len(scenario_reports), 3),
        },
    }
    aggregate["improvement_pct"] = {
        "average_queue_length_pct": pct_change(aggregate["baseline_fixed"]["average_queue_length"], aggregate["after_rl"]["average_queue_length"], higher_is_better=False),
        "average_waiting_time_pct": pct_change(aggregate["baseline_fixed"]["average_waiting_time"], aggregate["after_rl"]["average_waiting_time"], higher_is_better=False),
        "total_vehicles_processed_pct": pct_change(aggregate["baseline_fixed"]["total_vehicles_processed"], aggregate["after_rl"]["total_vehicles_processed"], higher_is_better=True),
        "total_reward_pct": pct_change(aggregate["baseline_fixed"]["total_reward"], aggregate["after_rl"]["total_reward"], higher_is_better=True),
    }

    report = {
        "max_steps": max_steps,
        "reward_mode": reward_mode,
        "decision_interval": decision_interval,
        "min_green_steps": min_green_steps,
        "scenario_reports": scenario_reports,
        "aggregate": aggregate,
    }

    comparison_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with comparison_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["scenario", "metric", "before_rl", "after_rl", "improvement_pct"])
        for item in scenario_reports:
            scenario_name = Path(item["config"]).name
            writer.writerow([
                scenario_name,
                "average_queue_length",
                item["baseline_fixed"]["average_queue_length"],
                item["after_rl"]["average_queue_length"],
                item["improvement_pct"]["average_queue_length_pct"],
            ])
            writer.writerow([
                scenario_name,
                "average_waiting_time",
                item["baseline_fixed"]["average_waiting_time"],
                item["after_rl"]["average_waiting_time"],
                item["improvement_pct"]["average_waiting_time_pct"],
            ])
            writer.writerow([
                scenario_name,
                "total_vehicles_processed",
                item["baseline_fixed"]["total_vehicles_processed"],
                item["after_rl"]["total_vehicles_processed"],
                item["improvement_pct"]["total_vehicles_processed_pct"],
            ])
            writer.writerow([
                scenario_name,
                "total_reward",
                item["baseline_fixed"]["total_reward"],
                item["after_rl"]["total_reward"],
                item["improvement_pct"]["total_reward_pct"],
            ])
        writer.writerow([
            "aggregate",
            "average_queue_length",
            aggregate["baseline_fixed"]["average_queue_length"],
            aggregate["after_rl"]["average_queue_length"],
            aggregate["improvement_pct"]["average_queue_length_pct"],
        ])
        writer.writerow([
            "aggregate",
            "average_waiting_time",
            aggregate["baseline_fixed"]["average_waiting_time"],
            aggregate["after_rl"]["average_waiting_time"],
            aggregate["improvement_pct"]["average_waiting_time_pct"],
        ])
        writer.writerow([
            "aggregate",
            "total_vehicles_processed",
            aggregate["baseline_fixed"]["total_vehicles_processed"],
            aggregate["after_rl"]["total_vehicles_processed"],
            aggregate["improvement_pct"]["total_vehicles_processed_pct"],
        ])
        writer.writerow([
            "aggregate",
            "total_reward",
            aggregate["baseline_fixed"]["total_reward"],
            aggregate["after_rl"]["total_reward"],
            aggregate["improvement_pct"]["total_reward_pct"],
        ])

    print(f"Saved comparison JSON: {comparison_json}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fixed policy vs trained Q-learning policy.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["traffic_low.sumocfg", "traffic_medium.sumocfg", "traffic_heavy.sumocfg"],
        help="One or more SUMO config paths to benchmark",
    )
    parser.add_argument("--max-steps", type=int, default=600, help="Simulation steps for each policy")
    parser.add_argument("--model", default="models/q_table.pkl", help="Path to trained Q-table")
    parser.add_argument("--use-gui", action="store_true", help="Run SUMO with GUI")
    parser.add_argument("--delay", type=int, default=100, help="GUI delay in milliseconds")
    parser.add_argument("--decision-interval", type=int, default=10, help="Simulation steps executed per agent decision")
    parser.add_argument("--min-green-steps", type=int, default=20, help="Minimum green hold before a switch is allowed")
    parser.add_argument("--switch-penalty", type=float, default=4.0, help="Reward penalty applied when phase switches")
    parser.add_argument("--reward-wait-weight", type=float, default=0.02, help="Waiting-time contribution weight in hybrid reward")
    parser.add_argument(
        "--reward-mode",
        choices=["queue_length", "waiting_time", "hybrid"],
        default="hybrid",
        help="Reward shaping used during benchmarking",
    )

    args = parser.parse_args()

    evaluate_many(
        config_paths=args.configs,
        max_steps=args.max_steps,
        model_path=args.model,
        use_gui=args.use_gui,
        gui_delay=args.delay,
        decision_interval=args.decision_interval,
        min_green_steps=args.min_green_steps,
        switch_penalty=args.switch_penalty,
        reward_wait_weight=args.reward_wait_weight,
        reward_mode=args.reward_mode,
    )
