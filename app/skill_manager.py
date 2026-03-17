from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class SkillRegistryError(Exception):
    """Raised when a problem occurs while loading skills."""


class SkillNotFound(SkillRegistryError):
    """Raised when a requested skill is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Skill '{name}' was not found.")


class SkillDisabled(SkillRegistryError):
    """Raised when a requested skill is disabled."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Skill '{name}' is disabled.")


class SkillUnavailable(SkillRegistryError):
    """Raised when a skill is missing required runtime dependencies."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"Skill '{name}' is unavailable: {reason}")


class SkillManager:
    """Loads OpenClaw-style skill packs and exposes execution helpers."""

    def __init__(self, packs_dir: Optional[str] = None, state_file: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parent
        self.packs_dir = Path(packs_dir or (root.parent / "skills"))
        default_state = root.parent / "data" / "skills_state.json"
        self.state_file = Path(state_file or default_state)
        self.skills: Dict[str, SkillPack] = {}
        self.enabled_state: Dict[str, bool] = {}
        self._load_state()
        self.reload()

    def reload(self) -> None:
        """Reloads all skill packs under skills/."""
        self.skills.clear()
        if not self.packs_dir.exists():
            raise SkillRegistryError(f"Skill packs directory not found: {self.packs_dir}")

        for pack_dir in sorted(self.packs_dir.iterdir()):
            if not pack_dir.is_dir():
                continue
            if pack_dir.name.startswith("_"):
                continue
            if pack_dir.name == "__pycache__":
                continue

            skill_md = pack_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            parsed = _parse_skill_markdown(skill_md)
            command_template = _resolve_command_template(parsed, pack_dir)

            skill_name = parsed.get("name") or pack_dir.name
            description = parsed.get("description") or f"Skill pack: {skill_name}"
            when_to_use = parsed.get("when_to_use") or ""

            self.skills[skill_name] = SkillPack(
                name=skill_name,
                description=description,
                when_to_use=when_to_use,
                pack_dir=pack_dir,
                skill_md_path=skill_md,
                command_template=command_template,
                metadata=parsed.get("metadata"),
                enabled=self.enabled_state.get(skill_name, True),
            )
        self._prune_state()
        self._save_state()

    def list_skills(self, include_disabled: bool = True) -> List["SkillPack"]:
        skills = list(self.skills.values())
        if include_disabled:
            return skills
        return [skill for skill in skills if skill.enabled]

    def get(self, name: str) -> "SkillPack":
        try:
            return self.skills[name]
        except KeyError as exc:
            raise SkillNotFound(name) from exc

    def execute(self, name: str, args: Any = "", session: Optional[any] = None) -> str:
        skill = self.get(name)
        if not skill.enabled:
            raise SkillDisabled(name)
        availability = skill.availability()
        if not availability["available"]:
            missing_env = availability["missing_env"]
            missing_bins = availability["missing_bins"]
            reasons: list[str] = []
            if missing_env:
                reasons.append(f"missing env: {', '.join(missing_env)}")
            if missing_bins:
                reasons.append(f"missing bins: {', '.join(missing_bins)}")
            raise SkillUnavailable(name, "; ".join(reasons) or "requirements not met")
        return skill.run(args=args, session=session)

    def names(self, include_disabled: bool = False) -> Iterable[str]:
        if include_disabled:
            return self.skills.keys()
        return (skill.name for skill in self.list_skills(include_disabled=False))

    def router_catalog(self) -> str:
        enabled_skills = self.list_skills(include_disabled=False)
        if not enabled_skills:
            return "(none)"
        lines: list[str] = []
        for skill in enabled_skills:
            parts = [f"- {skill.name}: {skill.description}"]
            when = getattr(skill, "when_to_use", "").strip()
            if when:
                parts.append(f"when: {when}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def available_skills_catalog(self) -> str:
        enabled_skills = self.list_skills(include_disabled=False)
        if not enabled_skills:
            return "<available_skills>\n(none)\n</available_skills>"

        lines = ["<available_skills>"]
        for skill in enabled_skills:
            if skill.disable_model_invocation:
                continue
            lines.append(f"- name: {skill.name}")
            lines.append(f"  path: {skill.skill_md_path}")
            lines.append(f"  description: {skill.description}")
            when = (skill.when_to_use or "").strip()
            if when:
                lines.append(f"  when_to_use: {when}")
            if skill.user_invocable is not None:
                lines.append(f"  user_invocable: {str(skill.user_invocable).lower()}")
            if skill.primary_env:
                lines.append(f"  primary_env: {skill.primary_env}")
            if skill.required_env:
                lines.append(f"  required_env: {', '.join(skill.required_env)}")
            if skill.required_bins:
                lines.append(f"  required_bins: {', '.join(skill.required_bins)}")
            lines.append(
                f"  invocation: {'packaged_runtime' if skill.has_runtime else 'manual_only'}"
            )
        lines.append("</available_skills>")
        return "\n".join(lines)

    def as_dicts(self) -> List[dict[str, str | list[str]]]:
        return [skill.to_dict() for skill in self.list_skills(include_disabled=True)]

    def read_skill_doc(self, name: str) -> str:
        skill = self.get(name)
        return skill.render_markdown()

    def set_enabled(self, name: str, enabled: bool) -> "SkillPack":
        skill = self.get(name)
        skill.enabled = bool(enabled)
        self.enabled_state[skill.name] = skill.enabled
        self._save_state()
        return skill

    def _load_state(self) -> None:
        if not self.state_file.exists():
            self.enabled_state = {}
            return
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            self.enabled_state = {}
            return
        state = payload.get("enabled")
        if isinstance(state, dict):
            self.enabled_state = {
                str(name): bool(value)
                for name, value in state.items()
                if isinstance(name, str) and name.strip()
            }
            return
        self.enabled_state = {}

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": {
                name: bool(value)
                for name, value in sorted(self.enabled_state.items())
            }
        }
        self.state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _prune_state(self) -> None:
        known = set(self.skills.keys())
        self.enabled_state = {
            name: value for name, value in self.enabled_state.items() if name in known
        }
        for name, skill in self.skills.items():
            self.enabled_state[name] = bool(skill.enabled)


class SkillPack:
    def __init__(
        self,
        name: str,
        description: str,
        when_to_use: str,
        pack_dir: Path,
        skill_md_path: Path,
        command_template: Optional[List[str]],
        metadata: Any = None,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self.when_to_use = when_to_use
        self.pack_dir = pack_dir
        self.skill_md_path = skill_md_path
        self.command_template = command_template or []
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.enabled = enabled

    def run(self, args: Any = "", session=None) -> str:
        if not self.has_runtime:
            raise SkillUnavailable(
                self.name,
                "no packaged runtime; read SKILL.md and use low-level tools instead",
            )
        timeout = int(os.environ.get("SKILL_SCRIPT_TIMEOUT", "30"))
        session_id = ""
        if session is not None and hasattr(session, "id"):
            session_id = str(getattr(session, "id", "") or "")
        context = {
            "input": _serialize_skill_input(args),
            "session_id": session_id,
            "pack_dir": str(self.pack_dir),
            "python": sys.executable,
        }
        cmd = [token.format_map(_SafeFormatDict(context)) for token in self.command_template]
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

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "pack_dir": str(self.pack_dir),
            "skill_md_path": str(self.skill_md_path),
            "skill_md_content": self.skill_md_path.read_text(encoding="utf-8"),
            "command_template": self.command_template,
            "has_runtime": self.has_runtime,
            "metadata": self.metadata,
            "user_invocable": self.user_invocable,
            "disable_model_invocation": self.disable_model_invocation,
            "required_env": self.required_env,
            "required_bins": self.required_bins,
            "primary_env": self.primary_env,
            "availability": self.availability(),
            "enabled": self.enabled,
        }

    def render_markdown(self) -> str:
        raw = self.skill_md_path.read_text(encoding="utf-8")
        return _strip_frontmatter(raw).strip()

    @property
    def has_runtime(self) -> bool:
        return bool(self.command_template)

    def availability(self) -> dict[str, Any]:
        missing_env = [name for name in self.required_env if not os.environ.get(name)]
        missing_bins = [name for name in self.required_bins if shutil.which(name) is None]
        return {
            "available": not missing_env and not missing_bins,
            "missing_env": missing_env,
            "missing_bins": missing_bins,
        }

    @property
    def openclaw_meta(self) -> dict[str, Any]:
        value = self.metadata.get("openclaw")
        return value if isinstance(value, dict) else {}

    @property
    def user_invocable(self) -> Optional[bool]:
        value = self.metadata.get("user-invocable")
        if isinstance(value, bool):
            return value
        value = self.openclaw_meta.get("user-invocable")
        return value if isinstance(value, bool) else None

    @property
    def disable_model_invocation(self) -> bool:
        value = self.metadata.get("disable-model-invocation")
        if isinstance(value, bool):
            return value
        value = self.openclaw_meta.get("disable-model-invocation")
        return bool(value) if isinstance(value, bool) else False

    @property
    def required_env(self) -> list[str]:
        requires = self.openclaw_meta.get("requires")
        if not isinstance(requires, dict):
            return []
        env = requires.get("env")
        if isinstance(env, list):
            return [str(item).strip() for item in env if str(item).strip()]
        return []

    @property
    def required_bins(self) -> list[str]:
        requires = self.openclaw_meta.get("requires")
        if not isinstance(requires, dict):
            return []
        bins = requires.get("bins")
        if isinstance(bins, list):
            return [str(item).strip() for item in bins if str(item).strip()]
        return []

    @property
    def primary_env(self) -> str:
        value = self.openclaw_meta.get("primaryEnv")
        return str(value).strip() if value is not None else ""


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _serialize_skill_input(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0:
        return text
    return text[end + 5 :]


def _parse_skill_markdown(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    lines = body.splitlines()
    name = _clean_scalar(frontmatter.get("name", ""))
    description = _clean_scalar(frontmatter.get("description", ""))
    when_to_use = _extract_when_to_use(lines)
    command = _coerce_command(frontmatter.get("command"))
    invocation = _clean_scalar(frontmatter.get("invocation", ""))

    if not name or not description:
        legacy = _parse_legacy_body(lines, path)
        if not name:
            name = legacy.get("name", "")
        if not description:
            description = legacy.get("description", "")
        if not when_to_use:
            when_to_use = legacy.get("when_to_use", "")
        if not command:
            command = legacy.get("command", [])

    return {
        "name": name,
        "description": description,
        "when_to_use": when_to_use,
        "command": command,
        "invocation": invocation,
        "metadata": frontmatter.get("metadata"),
    }


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    raw_frontmatter = text[4:end]
    body = text[end + 5 :]
    return _parse_frontmatter_block(raw_frontmatter), body


def _parse_frontmatter_block(block: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for line in block.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        payload[key] = _parse_frontmatter_value(value)
    return payload


def _parse_frontmatter_value(value: str) -> Any:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned[0] in "{[":
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
    if cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        return cleaned[1:-1]
    if cleaned.lower() in {"true", "false"}:
        return cleaned.lower() == "true"
    return cleaned


def _clean_scalar(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_command(value: Any) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return []


def _parse_legacy_body(lines: list[str], path: Path) -> dict[str, Any]:
    name = ""
    description = ""
    when_to_use = ""
    command: list[str] = []

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
        if lowered.startswith("command:") and not command:
            raw_command = line.split(":", 1)[1].strip()
            if raw_command:
                command = _parse_command_metadata(raw_command, path)
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
        "command": command,
    }


def _extract_when_to_use(lines: list[str]) -> str:
    heading_pattern = re.compile(r"^##\s+(.+?)\s*$")
    current_heading = ""
    when_lines: list[str] = []
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            current_heading = match.group(1).strip().lower()
            continue
        if current_heading in {"when to use", "何时使用"}:
            raw = line.strip().lstrip("- ").strip()
            if raw:
                when_lines.append(raw)
    return " / ".join(when_lines[:2]) if when_lines else ""


def _extract_command_from_usage(text: str, path: Path) -> list[str]:
    section_pattern = re.compile(
        r"^##\s+(usage|命令行调用)\s*$([\s\S]*?)(?=^##\s+|\Z)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    match = section_pattern.search(text)
    if not match:
        return []

    section = match.group(2)
    codeblocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", section)
    for block in codeblocks:
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tokens = shlex.split(line)
            except ValueError:
                continue
            normalized = _normalize_usage_command(tokens, path)
            if normalized:
                return normalized
    return []


def _normalize_usage_command(tokens: list[str], path: Path) -> list[str]:
    pack_dir = path.parent
    normalized: list[str] = []
    for token in tokens:
        mapped = token
        stripped = token.strip("\"'")
        lowered = stripped.lower()
        if lowered in {"<json>", "<input>", "{input}", "<args>", "<query>"}:
            mapped = "{input}"
        elif "session_id" in lowered and "{" in stripped:
            mapped = "{session_id}"
        elif stripped.startswith("skills/") and "/scripts/" in stripped:
            candidate = pack_dir / Path(stripped).name
            alt = pack_dir / "scripts" / Path(stripped).name
            if alt.exists():
                mapped = str(Path("scripts") / Path(stripped).name)
            elif candidate.exists():
                mapped = Path(stripped).name
        elif stripped.startswith("scripts/"):
            mapped = stripped
        normalized.append(mapped)
    return normalized


def _parse_command_metadata(raw_command: str, path: Path) -> list[str]:
    try:
        parsed = json.loads(raw_command)
    except json.JSONDecodeError:
        try:
            return shlex.split(raw_command)
        except ValueError as exc:
            raise SkillRegistryError(f"Invalid Command metadata in {path}: {exc}") from exc

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise SkillRegistryError(
            f"Invalid Command metadata in {path}: expected a JSON string array."
        )
    return parsed


def _resolve_command_template(parsed: dict[str, Any], pack_dir: Path) -> list[str]:
    invocation = str(parsed.get("invocation", "")).strip().lower()
    if invocation == "manual_only":
        return []

    command = parsed.get("command")
    if isinstance(command, list) and command:
        return command

    legacy_script = pack_dir / "scripts" / "run.py"
    if legacy_script.exists():
        return [
            "{python}",
            str(legacy_script),
            "--input",
            "{input}",
            "--session-id",
            "{session_id}",
        ]

    return []
