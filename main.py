from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Wind Agent entrypoint")
    parser.add_argument("mode", nargs="?", default="web", choices=["web", "cli"])
    args = parser.parse_args()

    if args.mode == "cli":
        from app.cli import main as cli_main

        cli_main()
        return

    from app.server import app

    app.run(port=5000)


if __name__ == "__main__":
    main()
