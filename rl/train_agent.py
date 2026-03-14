import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.q_learning_agent import QLearningAgent
from rl.traffic_env import TrafficEnv


def train(
    config_paths,
    episodes,
    max_steps,
    use_gui,
    gui_delay,
    learning_rate,
    discount_factor,
    epsilon,
    epsilon_min,
    epsilon_decay,
    bin_size,
    max_queue,
    seed,
    decision_interval,
    min_green_steps,
    switch_penalty,
    reward_wait_weight,
    reward_mode,
):
    random.seed(seed)
    np.random.seed(seed)

    resolved_configs = []
    for config_path in config_paths:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (PROJECT_ROOT / config_file).resolve()
        resolved_configs.append(config_file)

    model_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "q_table.pkl"
    rewards_path = results_dir / "training_rewards.csv"

    agent = QLearningAgent(
        action_size=2,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        bin_size=bin_size,
        max_queue=max_queue,
        seed=seed,
    )

    with rewards_path.open("w", newline="", encoding="utf-8") as rewards_file:
        writer = csv.writer(rewards_file)
        writer.writerow(["episode", "total_reward"])

        for episode in range(episodes):
            config_file = random.choice(resolved_configs)
            env = TrafficEnv(
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

            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            current_action = 0

            for _ in range(max_steps):
                action = agent.select_action(state, current_action=current_action)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                applied_action = int(info["applied_action"])

                agent.update(
                    state,
                    applied_action,
                    reward,
                    next_state,
                    done,
                    current_action=current_action,
                    next_action_context=applied_action,
                )
                total_reward += float(reward)
                state = next_state
                current_action = applied_action

                if done:
                    break

            env.close()
            agent.decay_exploration()

            writer.writerow([episode + 1, round(total_reward, 3)])
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Scenario: {config_file.name} | "
                f"Total Reward: {total_reward:.3f} | Epsilon: {agent.epsilon:.4f}"
            )

    agent.save(model_path)

    print(f"Saved Q-table: {model_path}")
    print(f"Saved reward trace: {rewards_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Q-learning traffic signal agent.")
    parser.add_argument(
        "--train-configs",
        nargs="+",
        default=["traffic_low.sumocfg", "traffic_medium.sumocfg", "traffic_heavy.sumocfg"],
        help="One or more SUMO config paths used for mixed-scenario training",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=600, help="Max simulation steps per episode")
    parser.add_argument("--use-gui", action="store_true", help="Run SUMO with GUI")
    parser.add_argument("--delay", type=int, default=100, help="GUI delay in milliseconds")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Q-learning alpha")
    parser.add_argument("--discount-factor", type=float, default=0.95, help="Q-learning gamma")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.99, help="Exploration decay per episode")
    parser.add_argument("--bin-size", type=int, default=4, help="Queue discretization bucket size")
    parser.add_argument("--max-queue", type=int, default=120, help="Queue clamp value before discretization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--decision-interval", type=int, default=10, help="Simulation steps executed per agent decision")
    parser.add_argument("--min-green-steps", type=int, default=20, help="Minimum green hold before a switch is allowed")
    parser.add_argument("--switch-penalty", type=float, default=4.0, help="Reward penalty applied when phase switches")
    parser.add_argument("--reward-wait-weight", type=float, default=0.02, help="Waiting-time contribution weight in hybrid reward")
    parser.add_argument(
        "--reward-mode",
        choices=["queue_length", "waiting_time", "hybrid"],
        default="hybrid",
        help="Reward shaping used during training",
    )

    args = parser.parse_args()

    train(
        config_paths=args.train_configs,
        episodes=args.episodes,
        max_steps=args.max_steps,
        use_gui=args.use_gui,
        gui_delay=args.delay,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        bin_size=args.bin_size,
        max_queue=args.max_queue,
        seed=args.seed,
        decision_interval=args.decision_interval,
        min_green_steps=args.min_green_steps,
        switch_penalty=args.switch_penalty,
        reward_wait_weight=args.reward_wait_weight,
        reward_mode=args.reward_mode,
    )
