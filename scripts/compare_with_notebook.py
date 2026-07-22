#!/usr/bin/env python3
"""Compare exported CSV metrics with notebook reference outputs.

Populate a reference CSV after running the research notebook, then extend this
script with study-specific tolerances. It is intentionally not part of runtime.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    candidate = pd.read_csv(args.candidate)
    common = [column for column in reference.columns if column in candidate.columns]
    print(reference[common].compare(candidate[common], keep_equal=False))


if __name__ == "__main__":
    main()

