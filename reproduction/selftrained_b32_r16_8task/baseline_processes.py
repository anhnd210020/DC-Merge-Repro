#!/usr/bin/env python3
"""Identify the one top-level official baseline evaluator, excluding its workers."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/selftrained_b32_r16_8task")
EXPECTED_FLAGS = {
    "--config": "vitB32_r16_8task",
    "--method": "dc_merge",
    "--smoothing": "linear",
}


def read_process(pid):
    proc = Path("/proc") / str(pid)
    try:
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        argv = [value.decode(errors="replace") for value in argv if value]
        stat = (proc / "stat").read_text().split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    if len(argv) < 2 or not Path(argv[0]).name.startswith("python"):
        return None
    if Path(argv[1]).name != "eval.py":
        return None
    for flag, expected in EXPECTED_FLAGS.items():
        try:
            if argv[argv.index(flag) + 1] != expected:
                return None
        except (ValueError, IndexError):
            return None
    return {
        "pid": pid,
        "ppid": int(stat[3]),
        "start_ticks": int(stat[21]),
        "argv": argv,
        "command_line": " ".join(argv),
    }


def matches():
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            process = read_process(int(entry.name))
            if process:
                found.append(process)
    return sorted(found, key=lambda item: item["pid"])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--all", action="store_true", help="include worker descendants")
    args = parser.parse_args()
    found = matches()
    matching_pids = {item["pid"] for item in found}
    top_level = [item for item in found if item["ppid"] not in matching_pids]
    selected = found if args.all else top_level
    print(json.dumps(selected, indent=2))
    if args.save:
        if len(top_level) != 1:
            raise SystemExit(
                f"refusing to guess: found {len(top_level)} top-level matching evaluators"
            )
        record = dict(top_level[0])
        record["detected_at"] = datetime.now(timezone.utc).isoformat()
        write_json(ROOT / "state" / "baseline_process.json", record)
        (ROOT / "state" / "baseline.pid").write_text(str(record["pid"]) + "\n")
        (ROOT / "state" / "baseline.command").write_text(record["command_line"] + "\n")


if __name__ == "__main__":
    main()
