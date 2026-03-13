import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.traffic_env import TrafficEnv


def action_to_label(action):
    return "NS_GREEN" if action == 0 else "EW_GREEN"


def run_day4_outputs(config_path, steps, use_gui, delay, output_dir):
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT / config_file).resolve()

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    state_trace_path = output_path / "rl_state_trace.csv"
    action_trace_path = output_path / "rl_action_trace.csv"
    reward_trace_path = output_path / "rl_reward_trace.csv"
    summary_path = output_path / "rl_env_summary.json"

    env = TrafficEnv(
        config_path=str(config_file),
        max_steps=steps,
        use_gui=use_gui,
        gui_delay_ms=delay,
        reward_mode="queue_length",
    )

    queue_sum = 0.0
    waiting_sum = 0.0
    total_vehicles_processed = 0

    state, _ = env.reset()

    with state_trace_path.open("w", newline="", encoding="utf-8") as state_file, \
        action_trace_path.open("w", newline="", encoding="utf-8") as action_file, \
        reward_trace_path.open("w", newline="", encoding="utf-8") as reward_file:

        state_writer = csv.writer(state_file)
        action_writer = csv.writer(action_file)
        reward_writer = csv.writer(reward_file)

        state_writer.writerow(["step", "north_queue", "south_queue", "east_queue", "west_queue"])
        action_writer.writerow(["step", "action"])
        reward_writer.writerow(["step", "reward"])

        for step in range(steps):
            action = 0 if (step // 30) % 2 == 0 else 1
            action_writer.writerow([step, action_to_label(action)])

            next_state, reward, _, truncated, info = env.step(action)

            state_writer.writerow(
                [
                    step,
                    int(next_state[0]),
                    int(next_state[1]),
                    int(next_state[2]),
                    int(next_state[3]),
                ]
            )
            reward_writer.writerow([step, round(float(reward), 3)])

            queue_sum += float(info["queue_length_total"])
            waiting_sum += float(info["waiting_time_total"])
            total_vehicles_processed += int(info["arrived_vehicles_step"])

            state = next_state
            if truncated:
                break

    env.close()

    effective_steps = step + 1 if steps > 0 else 0
    summary = {
        "total_steps": effective_steps,
        "average_queue_length": round(queue_sum / effective_steps, 3) if effective_steps else 0.0,
        "average_waiting_time": round(waiting_sum / effective_steps, 3) if effective_steps else 0.0,
        "total_vehicles_processed": total_vehicles_processed,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Generated: {state_trace_path}")
    print(f"Generated: {action_trace_path}")
    print(f"Generated: {reward_trace_path}")
    print(f"Generated: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Day-4 RL output files in required schema.")
    parser.add_argument("--config", default="sumo_env/config/traffic.sumocfg", help="SUMO config path")
    parser.add_argument("--steps", type=int, default=600, help="Number of simulation steps")
    parser.add_argument("--use-gui", action="store_true", help="Run SUMO with GUI")
    parser.add_argument("--delay", type=int, default=100, help="GUI delay ms")
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    args = parser.parse_args()

    run_day4_outputs(
        config_path=args.config,
        steps=args.steps,
        use_gui=args.use_gui,
        delay=args.delay,
        output_dir=args.output_dir,
    )
