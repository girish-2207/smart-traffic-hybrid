import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidate = (PROJECT_ROOT / path).resolve()
    if candidate.exists():
        return candidate

    return path.resolve()


class TrafficEnv(gym.Env):
    """Gym-style SUMO traffic signal environment."""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        config_path="sumo_env/config/traffic.sumocfg",
        max_steps=1200,
        use_gui=False,
        gui_delay_ms=100,
        tls_id=None,
        reward_mode="waiting_time",
        yellow_steps=3,
    ):
        super().__init__()
        self.config_path = str(resolve_project_path(config_path))
        self.max_steps = max_steps
        self.use_gui = use_gui
        self.gui_delay_ms = gui_delay_ms
        self.tls_id = tls_id
        self.reward_mode = reward_mode
        self.yellow_steps = yellow_steps

        self.step_count = 0
        self._incoming_lanes = []
        self._direction_lanes = {"north": [], "south": [], "east": [], "west": []}
        self._is_started = False

        # Action 0: North-South green, Action 1: East-West green
        self.action_space = spaces.Discrete(2)

        # State vector: [north_queue, south_queue, east_queue, west_queue]
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()

        command = ["sumo-gui" if self.use_gui else "sumo", "-c", self.config_path]
        if self.use_gui:
            command.extend(["--start", "--delay", str(self.gui_delay_ms)])

        traci.start(command)
        self._is_started = True

        if self.tls_id is None:
            self.tls_id = self._detect_tls_id_from_config(self.config_path)

        self._incoming_lanes = self._get_tls_incoming_lanes(self.tls_id)
        self._direction_lanes = self._group_lanes_by_direction(self._incoming_lanes)

        self.step_count = 0
        self._apply_action_phase(0)
        self._advance_steps(1)

        return self.get_state(), {}

    def step(self, action):
        action = int(action)
        if action not in (0, 1):
            raise ValueError("Action must be 0 (NS green) or 1 (EW green).")

        self._apply_action_phase(action)
        self._advance_steps(1)
        self.step_count += 1

        state = self.get_state()
        reward = self.compute_reward()

        terminated = False
        truncated = self.step_count >= self.max_steps
        waiting_time_total = self.get_total_waiting_time()
        arrived_vehicles_step = traci.simulation.getArrivedNumber()
        info = {
            "tls_id": self.tls_id,
            "phase": traci.trafficlight.getPhase(self.tls_id),
            "queue_length_total": int(np.sum(state)),
            "waiting_time_total": float(waiting_time_total),
            "arrived_vehicles_step": int(arrived_vehicles_step),
        }

        return state, reward, terminated, truncated, info

    def get_state(self):
        north_queue = self._direction_queue("north")
        south_queue = self._direction_queue("south")
        east_queue = self._direction_queue("east")
        west_queue = self._direction_queue("west")

        return np.array([north_queue, south_queue, east_queue, west_queue], dtype=np.float32)

    def compute_reward(self):
        if self.reward_mode == "queue_length":
            queue_total = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self._incoming_lanes)
            return -float(queue_total)

        waiting_total = self.get_total_waiting_time()
        return -float(waiting_total)

    def get_total_waiting_time(self):
        return float(sum(traci.lane.getWaitingTime(lane_id) for lane_id in self._incoming_lanes))

    def close(self):
        if self._is_started:
            traci.close()
            self._is_started = False

    def _apply_action_phase(self, action):
        # Network phase mapping in intersection.net.xml:
        # 0: NS green, 1: NS yellow, 2: EW green, 3: EW yellow
        target_green_phase = 0 if action == 0 else 2
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        if current_phase == target_green_phase:
            return

        if target_green_phase == 0:
            traci.trafficlight.setPhase(self.tls_id, 3)
        else:
            traci.trafficlight.setPhase(self.tls_id, 1)
        self._advance_steps(self.yellow_steps)

        traci.trafficlight.setPhase(self.tls_id, target_green_phase)

    def _advance_steps(self, count):
        for _ in range(count):
            traci.simulationStep()

    def _direction_queue(self, direction):
        lanes = self._direction_lanes[direction]
        return float(sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lanes))

    @staticmethod
    def _get_tls_incoming_lanes(tls_id):
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        return sorted({link_group[0][0] for link_group in controlled_links if link_group})

    @staticmethod
    def _group_lanes_by_direction(lane_ids):
        grouped = {"north": [], "south": [], "east": [], "west": []}
        for lane_id in lane_ids:
            direction = lane_id.split("_", 1)[0]
            if direction in grouped:
                grouped[direction].append(lane_id)
        return grouped

    @staticmethod
    def _detect_tls_id_from_config(config_path):
        config_file = resolve_project_path(config_path)
        config_tree = ET.parse(config_file)
        config_root = config_tree.getroot()

        net_file_node = config_root.find("./input/net-file")
        if net_file_node is None:
            raise RuntimeError("No <net-file> entry found in SUMO config.")

        net_file_value = net_file_node.attrib.get("value")
        if not net_file_value:
            raise RuntimeError("The <net-file> entry has no value attribute.")

        net_path = (config_file.parent / net_file_value).resolve()
        net_tree = ET.parse(net_path)
        net_root = net_tree.getroot()

        for junction in net_root.findall("junction"):
            if junction.attrib.get("type") == "traffic_light":
                tls_id = junction.attrib.get("id")
                if tls_id:
                    return tls_id

        raise RuntimeError("No traffic_light junction found in network file.")
