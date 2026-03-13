import argparse
import json
from pathlib import Path

import traci


TLS_ID = "center"
PHASE_SEQUENCE = [0, 1, 2, 3]  # NS green, NS yellow, EW green, EW yellow
PHASE_DURATIONS = [40, 3, 40, 3]
STOP_SPEED_THRESHOLD = 0.1
STOP_DISTANCE_TO_JUNCTION = 12.0


def _set_next_phase(sequence_index):
    phase = PHASE_SEQUENCE[sequence_index]
    duration = PHASE_DURATIONS[sequence_index]
    traci.trafficlight.setPhase(TLS_ID, phase)
    return phase, duration


def run(cfg_path="traffic_medium.sumocfg", steps=1200, output_path="results/tls_verification.json"):
    sumo_cmd = ["sumo", "-c", cfg_path]

    phase_switches = 0
    phase_counts = {str(phase): 0 for phase in PHASE_SEQUENCE}
    red_observation_events = 0
    stopped_on_red_events = 0

    traci.start(sumo_cmd)
    try:
        sequence_index = 0
        active_phase, phase_remaining = _set_next_phase(sequence_index)
        phase_switches += 1

        for _ in range(steps):
            traci.simulationStep()

            phase_counts[str(active_phase)] += 1
            phase_remaining -= 1

            tls_state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
            controlled_links = traci.trafficlight.getControlledLinks(TLS_ID)

            for link_idx, signal_state in enumerate(tls_state):
                if signal_state.lower() != "r" or not controlled_links[link_idx]:
                    continue

                incoming_lane_id = controlled_links[link_idx][0][0]
                lane_length = traci.lane.getLength(incoming_lane_id)
                lane_vehicle_ids = traci.lane.getLastStepVehicleIDs(incoming_lane_id)

                saw_vehicle_near_stopline = False
                for veh_id in lane_vehicle_ids:
                    lane_pos = traci.vehicle.getLanePosition(veh_id)
                    distance_to_junction = lane_length - lane_pos
                    if distance_to_junction > STOP_DISTANCE_TO_JUNCTION:
                        continue

                    saw_vehicle_near_stopline = True
                    speed = traci.vehicle.getSpeed(veh_id)
                    if speed <= STOP_SPEED_THRESHOLD:
                        stopped_on_red_events += 1

                if saw_vehicle_near_stopline:
                    red_observation_events += 1

            if phase_remaining == 0:
                sequence_index = (sequence_index + 1) % len(PHASE_SEQUENCE)
                active_phase, phase_remaining = _set_next_phase(sequence_index)
                phase_switches += 1
    finally:
        traci.close()

    pass_switching = phase_switches >= 4
    pass_red_stop = stopped_on_red_events > 0

    report = {
        "config": cfg_path,
        "tls_id": TLS_ID,
        "steps": steps,
        "phase_sequence": PHASE_SEQUENCE,
        "phase_durations": PHASE_DURATIONS,
        "phase_switches": phase_switches,
        "phase_step_counts": phase_counts,
        "red_observation_events": red_observation_events,
        "stopped_on_red_events": stopped_on_red_events,
        "pass_switching": pass_switching,
        "pass_red_stop": pass_red_stop,
        "overall_pass": pass_switching and pass_red_stop,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"TLS verification report written to: {output_file}")
    print(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TLS switching and red-light stopping behavior.")
    parser.add_argument("--config", default="traffic_medium.sumocfg", help="SUMO config path")
    parser.add_argument("--steps", type=int, default=1200, help="Simulation steps")
    parser.add_argument("--output", default="results/tls_verification.json", help="Output JSON path")
    args = parser.parse_args()

    result = run(cfg_path=args.config, steps=args.steps, output_path=args.output)
    raise SystemExit(0 if result["overall_pass"] else 1)
