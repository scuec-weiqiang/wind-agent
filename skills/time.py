from __future__ import annotations

from datetime import datetime

from .base import Skill


class TimeSkill(Skill):
    name = "time"
    description = "Returns the current local time or formatted string."

    def run(self, args: str = "", session=None) -> str:
        fmt = args.strip() or "%Y-%m-%d %H:%M:%S"
        try:
            return datetime.now().strftime(fmt)
        except Exception:
            return f"Unable to format time with '{fmt}'."


skill = TimeSkill()
