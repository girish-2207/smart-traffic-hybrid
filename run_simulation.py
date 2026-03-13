import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import traci


def detect_tls_id_from_config(config_path):
    config_file = Path(config_path)
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


def get_tls_incoming_lanes(tls_id):
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    lane_ids = {link_group[0][0] for link_group in controlled_links if link_group}
    return sorted(lane_ids)


def get_tls_incoming_edges(lane_ids):
    return sorted({lane_id.rsplit("_", 1)[0] for lane_id in lane_ids})


def main():
    parser = argparse.ArgumentParser(description="Run SUMO simulation through Python TraCI control.")
    parser.add_argument("--config", default="sumo_env/config/traffic.sumocfg", help="Path to SUMO config")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--tls-id", default=None, help="Traffic light id (auto-detected if omitted)")
    parser.add_argument("--delay", type=int, default=150, help="GUI delay in milliseconds")
    parser.add_argument("--nogui", action="store_true", help="Run without SUMO GUI")
    parser.add_argument("--state-csv", default="results/rl_state_trace.csv", help="Path to state trace CSV output")
    parser.add_argument("--summary-json", default="results/rl_state_summary.json", help="Path to state summary JSON output")
    args = parser.parse_args()

    binary = "sumo" if args.nogui else "sumo-gui"
    sumo_cmd = [binary, "-c", args.config]

    if not args.nogui:
        sumo_cmd.extend(["--start", "--delay", str(args.delay)])

    tls_id = args.tls_id or detect_tls_id_from_config(args.config)

    traci.start(sumo_cmd)
    step = 0
    phase_switches = 0
    state_rows = []

    incoming_lanes = get_tls_incoming_lanes(tls_id)
    incoming_edges = get_tls_incoming_edges(incoming_lanes)

    try:
        while step < args.steps:
            # 0-29 seconds: NS green (phase 0), 30-59 seconds: EW green (phase 2), then repeat.
            phase_in_cycle = step % 60
            target_phase = 0 if phase_in_cycle < 30 else 2

            if traci.trafficlight.getPhase(tls_id) != target_phase:
                traci.trafficlight.setPhase(tls_id, target_phase)
                phase_switches += 1

            traci.simulationStep()

            lane_vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in incoming_lanes)
            lane_queue_length = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in incoming_lanes)
            lane_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in incoming_lanes)
            lane_occupancy_avg = (
                sum(traci.lane.getLastStepOccupancy(lane_id) for lane_id in incoming_lanes) / len(incoming_lanes)
                if incoming_lanes
                else 0.0
            )
            edge_vehicle_count = sum(traci.edge.getLastStepVehicleNumber(edge_id) for edge_id in incoming_edges)

            state_rows.append(
                {
                    "step": step,
                    "vehicle_count_lanes": lane_vehicle_count,
                    "vehicle_count_edges": edge_vehicle_count,
                    "queue_length": lane_queue_length,
                    "waiting_time": round(lane_waiting_time, 3),
                    "lane_occupancy_avg": round(lane_occupancy_avg, 3),
                    "phase": traci.trafficlight.getPhase(tls_id),
                }
            )

            step += 1
    finally:
        traci.close()

    state_csv_path = Path(args.state_csv)
    state_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with state_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(state_rows[0].keys()))
        writer.writeheader()
        writer.writerows(state_rows)

    summary = {
        "tls_id": tls_id,
        "steps": args.steps,
        "phase_switches": phase_switches,
        "incoming_lanes": incoming_lanes,
        "incoming_edges": incoming_edges,
        "max_vehicle_count_lanes": max(row["vehicle_count_lanes"] for row in state_rows),
        "max_queue_length": max(row["queue_length"] for row in state_rows),
        "max_waiting_time": max(row["waiting_time"] for row in state_rows),
        "max_lane_occupancy_avg": max(row["lane_occupancy_avg"] for row in state_rows),
        "state_csv": str(state_csv_path),
    }

    summary_json_path = Path(args.summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Detected TLS ID: {tls_id}")
    print(f"Simulation steps: {args.steps}")
    print(f"Phase switches applied by Python: {phase_switches}")
    print(f"State trace saved to: {state_csv_path}")
    print(f"State summary saved to: {summary_json_path}")


if __name__ == "__main__":
    main()
