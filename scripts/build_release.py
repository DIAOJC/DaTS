#!/usr/bin/env python3
"""Build public archives after the English/privacy audit passes."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dats.release import make_release

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'dist'))
    args = parser.parse_args()
    print(json.dumps(make_release(ROOT, args.output), indent=2))
