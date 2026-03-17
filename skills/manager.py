from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .base import Skill


class SkillRegistryError(Exception):
    """Raised when a problem occurs while loading skills."""


class SkillNotFound(SkillRegistryError):
    """Raised when a requested skill is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Skill '{name}' was not found.")


class SkillManager:
    """Loads OpenClaw-style skill packs and exposes execution helpers."""

    def __init__(self, packs_dir: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parent
        self.packs_dir = Path(packs_dir or (root / "packs"))
        self.skills: Dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        """Reloads all skill packs under skills/packs."""
        self.skills.clear()
        if not self.packs_dir.exists():
            raise SkillRegistryError(f"Skill packs directory not found: {self.packs_dir}")

        for pack_dir in sorted(self.packs_dir.iterdir()):
            if not pack_dir.is_dir():
                continue
            if pack_dir.name.startswith("_"):
                continue

            skill_md = pack_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            parsed = _parse_skill_markdown(skill_md)
            script_path = pack_dir / "scripts" / "run.py"
            if not script_path.exists():
                continue

            skill_name = parsed.get("name") or pack_dir.name
            description = parsed.get("description") or f"Skill pack: {skill_name}"
            when_to_use = parsed.get("when_to_use") or ""

            self.skills[skill_name] = SkillPack(
                name=skill_name,
                description=description,
                when_to_use=when_to_use,
                pack_dir=pack_dir,
                script_path=script_path,
            )

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

    def router_catalog(self) -> str:
        if not self.skills:
            return "(none)"
        lines: list[str] = []
        for skill in self.list_skills():
            when = getattr(skill, "when_to_use", "").strip()
            if when:
                lines.append(f"- {skill.name}: {skill.description} | when: {when}")
            else:
                lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)


class SkillPack(Skill):
    def __init__(
        self,
        name: str,
        description: str,
        when_to_use: str,
        pack_dir: Path,
        script_path: Path,
    ) -> None:
        self.name = name
        self.description = description
        self.when_to_use = when_to_use
        self.pack_dir = pack_dir
        self.script_path = script_path

    def run(self, args: str = "", session=None) -> str:
        timeout = int(os.environ.get("SKILL_SCRIPT_TIMEOUT", "30"))
        cmd = [sys.executable, str(self.script_path), "--input", args or ""]
        if session is not None:
            session_id = getattr(session, "id", "") if hasattr(session, "id") else ""
            if session_id:
                cmd.extend(["--session-id", str(session_id)])
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.pack_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Skill '{self.name}' timed out after {timeout}s."
        except Exception as exc:  # pragma: no cover
            return f"Skill '{self.name}' failed to run: {exc}"

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return stderr or stdout or f"Skill '{self.name}' failed with exit_code={proc.returncode}."
        return stdout or "(no output)"


def _parse_skill_markdown(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    name = ""
    description = ""
    when_to_use = ""

    for line in lines:
        if line.startswith("# ") and not name:
            name = line[2:].strip()
            continue
        lowered = line.lower()
        if lowered.startswith("name:") and not name:
            name = line.split(":", 1)[1].strip()
            continue
        if lowered.startswith("description:") and not description:
            description = line.split(":", 1)[1].strip()
            continue

    if not description:
        # First non-heading paragraph line as summary.
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("#"):
                continue
            if raw.startswith("-"):
                continue
            description = raw
            break

    heading_pattern = re.compile(r"^##\s+(.+?)\s*$")
    current_heading = ""
    when_lines: list[str] = []
    for line in lines:
        m = heading_pattern.match(line)
        if m:
            current_heading = m.group(1).strip().lower()
            continue
        if current_heading in {"when to use", "何时使用"}:
            raw = line.strip().lstrip("- ").strip()
            if raw:
                when_lines.append(raw)

    if when_lines:
        when_to_use = " / ".join(when_lines[:2])

    return {
        "name": name,
        "description": description,
        "when_to_use": when_to_use,
    }
