from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Skill(ABC):
    """Base interface for any skill that can be invoked from the agent."""

    name: str = "base"
    description: str = "A skill must provide a brief description."

    @abstractmethod
    def run(self, args: str = "", session: Optional[Any] = None) -> str:
        """Execute the skill with the given arguments and optional session context."""
        raise NotImplementedError("Skills must implement run().")
