from __future__ import annotations

import os
import sys
import textwrap
from typing import List

import requests

from app.chat import ChatSession, DEFAULT_MODEL, DEFAULT_URL
from skills.manager import SkillManager, SkillNotFound, SkillRegistryError

COMMAND_PREFIX = ":"

WELCOME_MESSAGE = textwrap.dedent(
    """\
    Simplified agent CLI (matches qwen3.5:9b on Ollama).
    Prefix commands with `:` (e.g. `:skills` or `:skill time`). Type `:help` for a quick list.
    Use OLLAMA_MODEL/OLLAMA_URL to override the defaults, and SYSTEM_PROMPT to steer the assistant.
    """
)


class AgentCLI:
    def __init__(self) -> None:
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        base_url = os.environ.get("OLLAMA_URL", DEFAULT_URL)
        system_prompt = os.environ.get(
            "SYSTEM_PROMPT",
            "You are a helpful assistant coordinating skills and answering questions.",
        )
        self.session = ChatSession(
            model=model, base_url=base_url, system_prompt=system_prompt
        )
        self.skill_manager: SkillManager | None = None
        self._load_skills()

    def _load_skills(self) -> None:
        try:
            self.skill_manager = SkillManager()
        except SkillRegistryError as exc:
            print(f"[skills] {exc}", file=sys.stderr)
            self.skill_manager = None

    def run(self) -> None:
        print(WELCOME_MESSAGE)
        while True:
            try:
                prompt = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            if not prompt:
                continue

            if prompt.startswith(COMMAND_PREFIX):
                should_exit = self._handle_command(prompt[1:].strip())
                if should_exit:
                    break
                continue

            print("AI:", end=" ", flush=True)
            try:
                self.session.stream_chat(prompt, on_chunk=self._print_chunk)
            except requests.RequestException as exc:
                print(f"\n[network error] {exc}")
            else:
                print()

    def _print_chunk(self, chunk: str) -> None:
        print(chunk, end="", flush=True)

    def _handle_command(self, command: str) -> bool:
        if not command:
            return False

        tokens = command.split()
        keyword = tokens[0].lower()

        if keyword in {"help", "?"}:
            self._print_help()
        elif keyword in {"exit", "quit"}:
            print("Bye.")
            return True
        elif keyword == "clear":
            self.session.reset()
            print("Conversation context cleared.")
        elif keyword == "history":
            limit = self._parse_limit(tokens[1:])
            self._print_history(limit)
        elif keyword == "skills":
            self._handle_skills_command(tokens[1:])
        elif keyword == "skill":
            self._run_skill(tokens[1:])
        else:
            print("Unknown command. Type :help for a list.")

        return False

    @staticmethod
    def _parse_limit(tokens: List[str]) -> int:
        if tokens and tokens[0].isdigit():
            return max(1, int(tokens[0]))
        return 6

    def _print_history(self, limit: int) -> None:
        data = list(self.session.history())
        if not data:
            print("[history] No messages yet.")
            return

        for entry in data[-limit:]:
            role = entry["role"].upper()
            print(f"{role:<10}: {entry['content']}")

    def _handle_skills_command(self, args: List[str]) -> None:
        if not self.skill_manager:
            print("No skills are currently loaded.")
            return

        if args and args[0].lower() == "reload":
            try:
                self.skill_manager.reload()
            except SkillRegistryError as exc:
                print(f"[skills] {exc}")
            else:
                print("Skills reloaded.")
            return

        skills = self.skill_manager.list_skills()
        if not skills:
            print("No skills available.")
            return

        for skill in skills:
            print(f"{skill.name:<12} {skill.description}")

    def _run_skill(self, args: List[str]) -> None:
        if not args:
            print("Usage: :skill <name> [input]")
            return

        if not self.skill_manager:
            print("Skills are unavailable.")
            return

        name, *rest = args
        payload = " ".join(rest).strip()
        try:
            result = self.skill_manager.execute(name, payload, session=self.session)
        except SkillNotFound:
            print(f"Skill '{name}' not found.")
            return
        except SkillRegistryError as exc:
            print(f"[skills] {exc}")
            return

        print(f"[{name}] {result}")

    @staticmethod
    def _print_help() -> None:
        print(
            "\n".join(
                (
                    "Commands:",
                    "  :help                Show this message.",
                    "  :history [n]        Show the last n messages (default 6).",
                    "  :clear               Reset conversation history.",
                    "  :skills              List registered skills.",
                    "  :skills reload       Reload fresh skill definitions.",
                    "  :skill <name> [args] Invoke a skill with the given arguments.",
                    "  :exit / :quit        Exit the CLI.",
                )
            )
        )


def main() -> None:
    AgentCLI().run()


if __name__ == "__main__":
    main()
