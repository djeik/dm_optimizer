#!/usr/bin/env python

"""
For a list of runs stored as one JSON array in a file, this tool calculates
the mean solution value found, and records that average into the file.
"""

from __future__ import print_function

import sys
import json

verbosity = 0

def log(priority, *args, **kwargs):
    if priority <= verbosity:
        print(*args, **kwargs)

def elog(priority, *args, **kwargs):
    log(priority, *args, file=sys.stderr, **kwargs)

def statistics(input_path, force=False):
    """ Load the JSON from input_path and calculate statistics for the set of
        runs recorded. The statistics are recorded into the JSON file under
        'statistics'. If the `force` parameter is True, then statistics that
        have already been calculated and recorded will be recalculated.
        """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except IOError, OSError:
        print("Could not read file:", input_path, file=sys.stderr)
        return

    if "statistics" not in data:
        data["statistics"] = {}

    stats = data["statistics"]

    try:
        run_data = data["run_data"]
    except KeyError:
        print("No `run_data` entry in the loaded data from", input_path,
                file=sys.stderr)
        return None

    # calculate average solution found
    if force or "average_solution" not in stats:
        elog(1, "Calculating average solution.",
                "(forced)" if force and "average_solution" in stats else "")
        stats["average_solution"] = sum(d["fun"] for d in run_data) / len(run_data)

    try:
        s = json.dumps(data)
    except TypeError as e:
        print("JSON serialization failed:", e, file=sys.stderr)
        return None

    try:
        with open(input_path, 'w') as f:
            f.write(s)
    except OSError, IOError:
        print("Could not write file:", input_path)
        return None

if __name__ == "__main__":
    input_paths = sys.argv[1:]

    if not input_paths:
        print("No input paths given. Please supply at least one path.",
                file=sys.stderr)
        sys.exit(1)

    for p in input_paths:
        statistics(p)
