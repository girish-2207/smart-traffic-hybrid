import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "rl_state_trace.csv",
    "rl_action_trace.csv",
    "rl_reward_trace.csv",
    "rl_env_summary.json",
]

EXPECTED_STATE_HEADER = ["step", "north_queue", "south_queue", "east_queue", "west_queue"]
EXPECTED_ACTION_HEADER = ["step", "action"]
EXPECTED_REWARD_HEADER = ["step", "reward"]
EXPECTED_SUMMARY_KEYS = [
    "total_steps",
    "average_queue_length",
    "average_waiting_time",
    "total_vehicles_processed",
]
VALID_ACTIONS = {"NS_GREEN", "EW_GREEN"}


def read_csv_rows(file_path):
    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def verify_outputs(output_dir):
    results = []
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    # 1) Existence checks
    missing = []
    for name in REQUIRED_FILES:
        if not (output_path / name).exists():
            missing.append(name)
    if missing:
        for name in missing:
            results.append((False, f"Missing file: {name}"))
        return results

    state_path = output_path / "rl_state_trace.csv"
    action_path = output_path / "rl_action_trace.csv"
    reward_path = output_path / "rl_reward_trace.csv"
    summary_path = output_path / "rl_env_summary.json"

    # 2) Header checks
    state_header, state_rows = read_csv_rows(state_path)
    action_header, action_rows = read_csv_rows(action_path)
    reward_header, reward_rows = read_csv_rows(reward_path)

    results.append((state_header == EXPECTED_STATE_HEADER, f"State header: {state_header}"))
    results.append((action_header == EXPECTED_ACTION_HEADER, f"Action header: {action_header}"))
    results.append((reward_header == EXPECTED_REWARD_HEADER, f"Reward header: {reward_header}"))

    # 3) Row count checks
    state_n = len(state_rows)
    action_n = len(action_rows)
    reward_n = len(reward_rows)
    same_count = state_n == action_n == reward_n
    results.append((state_n > 0, f"State rows: {state_n}"))
    results.append((same_count, f"Row counts (state/action/reward): {state_n}/{action_n}/{reward_n}"))

    # 4) Action values and reward numeric/<=0 checks
    valid_actions = True
    for row in action_rows:
        if len(row) != 2 or row[1] not in VALID_ACTIONS:
            valid_actions = False
            break
    results.append((valid_actions, "Action values are NS_GREEN/EW_GREEN"))

    reward_values_ok = True
    for row in reward_rows:
        if len(row) != 2:
            reward_values_ok = False
            break
        try:
            value = float(row[1])
        except ValueError:
            reward_values_ok = False
            break
        if value > 0.0:
            reward_values_ok = False
            break
    results.append((reward_values_ok, "Reward values are numeric and <= 0"))

    # 5) Summary checks
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        results.append((False, "Summary JSON is invalid"))
        return results

    keys_ok = all(key in summary for key in EXPECTED_SUMMARY_KEYS)
    results.append((keys_ok, f"Summary keys present: {list(summary.keys())}"))

    if keys_ok:
        total_steps_ok = int(summary["total_steps"]) == state_n
        avg_queue_ok = float(summary["average_queue_length"]) >= 0.0
        avg_wait_ok = float(summary["average_waiting_time"]) >= 0.0
        processed_ok = int(summary["total_vehicles_processed"]) >= 0

        results.append((total_steps_ok, f"Summary total_steps matches rows: {summary['total_steps']}"))
        results.append((avg_queue_ok, f"average_queue_length: {summary['average_queue_length']}"))
        results.append((avg_wait_ok, f"average_waiting_time: {summary['average_waiting_time']}"))
        results.append((processed_ok, f"total_vehicles_processed: {summary['total_vehicles_processed']}"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify Day-4 RL output files and schema.")
    parser.add_argument("--output-dir", default="results", help="Directory containing Day-4 outputs")
    args = parser.parse_args()

    checks = verify_outputs(args.output_dir)
    failures = [msg for ok, msg in checks if not ok]

    print("Day-4 Output Verification")
    print("=" * 28)
    for ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {msg}")

    if failures:
        print("\nOverall: FAIL")
        raise SystemExit(1)

    print("\nOverall: PASS")


if __name__ == "__main__":
    main()
