import pickle
import random
from pathlib import Path

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for 2-phase traffic signal control."""

    def __init__(
        self,
        action_size=2,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        bin_size=2,
        max_queue=60,
        seed=42,
    ):
        self.action_size = int(action_size)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.bin_size = int(bin_size)
        self.max_queue = int(max_queue)

        self.q_table = {}
        random.seed(seed)
        np.random.seed(seed)

    def discretize_state(self, state):
        """Project lane queues into a compact axis-based state for better generalization."""
        north, south, east, west = [float(value) for value in state]
        ns_total = north + south
        ew_total = east + west
        imbalance = ns_total - ew_total

        return (
            self._bucket_queue(ns_total),
            self._bucket_queue(ew_total),
            self._bucket_imbalance(imbalance),
        )

    def _bucket_queue(self, value):
        clamped = max(0.0, min(float(value), float(self.max_queue)))
        return int(clamped // self.bin_size)

    def _bucket_imbalance(self, value):
        clamped = max(-float(self.max_queue), min(float(value), float(self.max_queue)))
        shifted = clamped + float(self.max_queue)
        return int(shifted // self.bin_size)

    def state_key(self, state, current_action=None):
        base_key = self.discretize_state(state)
        action_context = -1 if current_action is None else int(current_action)
        return base_key + (action_context,)

    def _ensure_state(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float32)

    def select_action(self, state, current_action=None):
        state_key = self.state_key(state, current_action=current_action)
        self._ensure_state(state_key)

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.q_table[state_key]
        return int(np.argmax(q_values))

    def select_greedy_action(self, state, current_action=None):
        """Select the best-known action without exploration."""
        state_key = self.state_key(state, current_action=current_action)
        self._ensure_state(state_key)
        q_values = self.q_table[state_key]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done, current_action=None, next_action_context=None):
        state_key = self.state_key(state, current_action=current_action)
        next_state_key = self.state_key(next_state, current_action=next_action_context)

        self._ensure_state(state_key)
        self._ensure_state(next_state_key)

        current_q = self.q_table[state_key][int(action)]
        next_max_q = float(np.max(self.q_table[next_state_key]))
        target_q = float(reward) + (0.0 if done else self.discount_factor * next_max_q)

        self.q_table[state_key][int(action)] = current_q + self.learning_rate * (target_q - current_q)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {k: v.tolist() for k, v in self.q_table.items()}
        payload = {
            "q_table": serializable,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "bin_size": self.bin_size,
            "max_queue": self.max_queue,
        }

        with model_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as file_obj:
            payload = pickle.load(file_obj)

        agent = cls(
            action_size=payload["action_size"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            bin_size=payload["bin_size"],
            max_queue=payload["max_queue"],
        )

        agent.q_table = {k: np.array(v, dtype=np.float32) for k, v in payload["q_table"].items()}
        return agent
