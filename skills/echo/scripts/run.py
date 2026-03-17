from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    text = (args.input or "").strip()
    print(text if text else "No text was provided to echo.")


if __name__ == "__main__":
    main()
