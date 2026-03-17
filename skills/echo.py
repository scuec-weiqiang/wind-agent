from __future__ import annotations

from .base import Skill


class EchoSkill(Skill):
    name = "echo"
    description = "Echoes the arguments back verbatim."

    def run(self, args: str = "", session=None) -> str:
        trimmed = args.strip()
        if not trimmed:
            return "No text was provided to echo."
        return trimmed


skill = EchoSkill()
