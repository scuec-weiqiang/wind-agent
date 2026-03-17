import importlib
import logging
import pkgutil
from typing import Dict, Iterable, List, Optional

from .base import Skill


class SkillRegistryError(Exception):
    """Raised when a problem occurs while loading skills."""


class SkillNotFound(SkillRegistryError):
    """Raised when a requested skill is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Skill '{name}' was not found.")


class SkillManager:
    """Loads skills from the skills package and exposes execution helpers."""

    def __init__(self, package_name: str = "skills") -> None:
        self.package_name = package_name
        self.skills: Dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        """Reloads all skill modules under the configured package."""
        self.skills.clear()
        try:
            package = importlib.import_module(self.package_name)
        except ImportError as exc:
            raise SkillRegistryError(
                f"Unable to import skill package '{self.package_name}'."
            ) from exc

        for finder, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if module_name.startswith("_"):
                continue

            full_module = f"{self.package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module)
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to load skill module %s: %s", full_module, exc)
                continue

            candidate = getattr(module, "skill", None)
            if isinstance(candidate, Skill):
                self.skills[candidate.name] = candidate

    def list_skills(self) -> List[Skill]:
        return list(self.skills.values())

    def get(self, name: str) -> Skill:
        try:
            return self.skills[name]
        except KeyError as exc:
            raise SkillNotFound(name) from exc

    def execute(self, name: str, args: str = "", session: Optional[any] = None) -> str:
        skill = self.get(name)
        return skill.run(args=args, session=session)

    def names(self) -> Iterable[str]:
        return self.skills.keys()
