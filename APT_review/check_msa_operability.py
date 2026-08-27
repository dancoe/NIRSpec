#!/usr/bin/env python3
"""Cross-match APT MSA Target Info exports with sparse operability maps."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

SOURCE_TYPES = {"Primary", "Filler", "Contaminant"}


def load_map(path):
    with Path(path).open(encoding="utf-8") as handle:
        return {
            (int(row["Q"]), int(row["x"]), int(row["y"])): row
            for row in json.load(handle)["msaoper"]
        }


def csv_columns(fieldnames):
    return {name.strip().upper(): name for name in fieldnames or []}


def pointing_from_name(path):
    match = re.search(r"(?:^|[ :])(ep\d+pt\d+)[^-]*-", path.name.lower())
    return match.group(1) if match else "unknown"


def iter_targets(exports_dir, pointing_filter=None):
    for path in sorted(Path(exports_dir).glob("*.csv")):
        pointing = pointing_from_name(path)
        if pointing_filter and pointing != pointing_filter:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = csv_columns(reader.fieldnames)
            required = {"ID", "SOURCE TYPE", "QUADRANT", "COLUMN (DISP)", "ROW (SPAT)"}
            if not required.issubset(columns):
                continue
            nod = "n2" if "n2" in path.name.lower() else "n1"
            for row in reader:
                source_type = row[columns["SOURCE TYPE"]].strip()
                if source_type not in SOURCE_TYPES:
                    continue
                coordinate = (
                    int(float(row[columns["QUADRANT"]])),
                    int(float(row[columns["COLUMN (DISP)"]])),
                    int(float(row[columns["ROW (SPAT)"]])),
                )
                yield {
                    "pointing": pointing,
                    "nod": nod,
                    "target_id": row[columns["ID"]].strip(),
                    "source_type": source_type,
                    "coordinate": coordinate,
                    "file": path.name,
                }


def format_coordinate(coordinate):
    quadrant, column, row = coordinate
    return f"q{quadrant}d{column}s{row}"


def report_matches(targets, current_map, previous_map=None):
    matches = defaultdict(set)
    for target in targets:
        coordinate = target["coordinate"]
        current = current_map.get(coordinate)
        if not current:
            continue
        current_state = current.get("Internal state", "").lower()
        previous_state = "unlisted"
        if previous_map is not None:
            previous = previous_map.get(coordinate)
            previous_state = previous.get("Internal state", "unlisted").lower() if previous else "unlisted"
        failed_closed = current_state == "closed"
        newly_failed_closed = previous_map is not None and failed_closed and previous_state != "closed"
        failed_open = current_state == "open"
        if failed_closed or failed_open:
            key = (
                target["pointing"],
                target["target_id"],
                target["source_type"],
                coordinate,
                current_state,
                previous_state,
            )
            matches[key].add(target["nod"])
            if newly_failed_closed:
                matches[key].add("newly-closed")
    return matches


def print_matches(title, matches):
    print(title)
    if not matches:
        print("  none")
        return
    for key, nods in sorted(matches.items(), key=lambda item: (item[0][0], item[0][2], int(item[0][1]))):
        pointing, target_id, source_type, coordinate, current_state, previous_state = key
        nod_labels = ",".join(sorted(nods))
        print(
            f"  {pointing} | target {target_id} | {source_type} | "
            f"{format_coordinate(coordinate)} | current={current_state} | "
            f"previous={previous_state} | {nod_labels}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports_dir", type=Path, help="Directory containing APT MSA Target Info CSV files")
    parser.add_argument("current_map", type=Path, help="Current MSA operability JSON file")
    parser.add_argument("--previous-map", type=Path, help="Previous/design MSA operability JSON for transition checks")
    parser.add_argument("--pointing", help="Restrict output to a pointing such as ep10pt2")
    args = parser.parse_args()

    targets = list(iter_targets(args.exports_dir, args.pointing))
    current_map = load_map(args.current_map)
    previous_map = load_map(args.previous_map) if args.previous_map else None
    matches = report_matches(targets, current_map, previous_map)
    print(f"Scanned {len(targets)} target rows and {len(current_map)} explicit map entries.")
    print_matches("Explicit failed-shutter assignments:", matches)
    if previous_map is not None:
        newly_closed = {key: nods for key, nods in matches.items() if "newly-closed" in nods}
        print_matches("Newly failed-closed assignments:", newly_closed)


if __name__ == "__main__":
    main()
