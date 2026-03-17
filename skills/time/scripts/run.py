from __future__ import annotations

import argparse
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    fmt = (args.input or "").strip() or "%Y-%m-%d %H:%M:%S"
    try:
        print(datetime.now().strftime(fmt))
    except Exception:
        print(f"Unable to format time with '{fmt}'.")


if __name__ == "__main__":
    main()
