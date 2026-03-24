from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class SkillRegistryError(Exception):
    """Raised when a problem occurs while loading skills."""


class SkillNotFound(SkillRegistryError):
    """Raised when a requested skill is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Skill '{name}' was not found.")


class SkillNotFoundError(SkillRegistryError):
    """Raised when no skill can be resolved for a requested name."""

    def __init__(self, requested_name: str, candidates: Optional[list[str]] = None) -> None:
        self.requested_name = requested_name
        self.candidates = candidates or []
        detail = (
            f"Skill '{requested_name}' was not found."
            if not self.candidates
            else f"Skill '{requested_name}' was not found. Did you mean: {', '.join(self.candidates)}"
        )
        super().__init__(detail)


class SkillAmbiguousError(SkillRegistryError):
    """Raised when the requested name matches more than one skill."""

    def __init__(self, requested_name: str, candidates: list[str]) -> None:
        self.requested_name = requested_name
        self.candidates = candidates
        super().__init__(
            f"Skill '{requested_name}' is ambiguous: {', '.join(candidates)}"
        )


class SkillDisabled(SkillRegistryError):
    """Raised when a requested skill is disabled."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Skill '{name}' is disabled.")


class SkillUnavailable(SkillRegistryError):
    """Raised when a skill is missing required runtime dependencies."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"Skill '{name}' is unavailable: {reason}")


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    display_name: str
    aliases: list[str]
    source_path: str


def normalize_skill_key(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    normalized = normalized.strip("-")
    if not normalized:
        raise ValueError("Skill key is empty after normalization.")
    return normalized


class SkillManager:
    """Loads OpenClaw-style skill packs and exposes execution helpers."""

    def __init__(self, packs_dir: Optional[str] = None, state_file: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parent
        self.packs_dir = Path(packs_dir or (root.parent / "skills"))
        default_state = root.parent / "data" / "skills_state.json"
        self.state_file = Path(state_file or default_state)
        self.skills: Dict[str, SkillPack] = {}
        self.alias_index: Dict[str, set[str]] = {}
        self.enabled_state: Dict[str, bool] = {}
        self._load_state()
        self.reload()

    def reload(self) -> None:
        """Reloads all skill packs under skills/."""
        self.skills.clear()
        self.alias_index.clear()
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
            display_name = parsed.get("name") or pack_dir.name
            skill_id = _resolve_skill_id(parsed, display_name)
            description = parsed.get("description") or f"Skill pack: {display_name}"
            when_to_use = parsed.get("when_to_use") or ""
            aliases = _build_skill_aliases(
                display_name=display_name,
                skill_id=skill_id,
                explicit_aliases=parsed.get("aliases"),
            )

            if skill_id in self.skills:
                existing = self.skills[skill_id]
                raise SkillRegistryError(
                    "Duplicate canonical skill_id detected: "
                    f"'{skill_id}' from '{pack_dir}' conflicts with '{existing.pack_dir}'."
                )

            self.skills[skill_id] = SkillPack(
                skill_id=skill_id,
                display_name=display_name,
                aliases=aliases,
                description=description,
                when_to_use=when_to_use,
                pack_dir=pack_dir,
                skill_md_path=skill_md,
                command_template=command_template,
                metadata=parsed.get("metadata"),
                enabled=self.enabled_state.get(skill_id, True),
            )

        for skill in self.skills.values():
            for alias in skill.aliases:
                self.alias_index.setdefault(alias, set()).add(skill.skill_id)
        self._prune_state()
        self._save_state()

    def list_skills(self, include_disabled: bool = True) -> List["SkillPack"]:
        skills = list(self.skills.values())
        if include_disabled:
            return skills
        return [skill for skill in skills if skill.enabled]

    def get(self, name: str) -> "SkillPack":
        skill, _ = self.resolve_skill(name)
        return skill

    def resolve_skill(self, requested_name: str) -> tuple["SkillPack", str]:
        requested_raw = str(requested_name or "").strip()
        if not requested_raw:
            raise SkillNotFoundError(requested_name, candidates=self._top_suggestions(""))

        if requested_raw in self.skills:
            return self.skills[requested_raw], "exact_id"

        direct_alias = self.alias_index.get(requested_raw, set())
        if len(direct_alias) == 1:
            resolved_id = next(iter(direct_alias))
            return self.skills[resolved_id], "exact_alias"
        if len(direct_alias) > 1:
            candidates = sorted(direct_alias)
            raise SkillAmbiguousError(requested_raw, candidates=candidates)

        try:
            normalized = normalize_skill_key(requested_raw)
        except ValueError:
            raise SkillNotFoundError(requested_raw, candidates=self._top_suggestions(requested_raw))

        normalized_matches: set[str] = set()
        for skill in self.skills.values():
            try:
                if normalize_skill_key(skill.skill_id) == normalized:
                    normalized_matches.add(skill.skill_id)
                    continue
            except ValueError:
                pass
            for alias in skill.aliases:
                try:
                    if normalize_skill_key(alias) == normalized:
                        normalized_matches.add(skill.skill_id)
                        break
                except ValueError:
                    continue

        if len(normalized_matches) == 1:
            resolved_id = next(iter(normalized_matches))
            return self.skills[resolved_id], "normalized_alias"
        if len(normalized_matches) > 1:
            candidates = sorted(normalized_matches)
            raise SkillAmbiguousError(requested_raw, candidates=candidates)

        raise SkillNotFoundError(
            requested_raw,
            candidates=self._top_suggestions(requested_raw),
        )

    def execute(self, name: str, args: Any = "", session: Optional[any] = None) -> str:
        skill, matched_by = self.resolve_skill(name)
        print(
            f"[skill-resolve] requested_name={name!r} "
            f"resolved_skill_id={skill.skill_id!r} matched_by={matched_by}",
            flush=True,
        )
        if not skill.enabled:
            raise SkillDisabled(skill.skill_id)
        availability = skill.availability()
        if not availability["available"]:
            missing_env = availability["missing_env"]
            missing_bins = availability["missing_bins"]
            reasons: list[str] = []
            if missing_env:
                reasons.append(f"missing env: {', '.join(missing_env)}")
            if missing_bins:
                reasons.append(f"missing bins: {', '.join(missing_bins)}")
            raise SkillUnavailable(skill.skill_id, "; ".join(reasons) or "requirements not met")
        return skill.run(args=args, session=session)

    def names(self, include_disabled: bool = False) -> Iterable[str]:
        if include_disabled:
            return self.skills.keys()
        return (skill.skill_id for skill in self.list_skills(include_disabled=False))

    def router_catalog(self) -> str:
        enabled_skills = self.list_skills(include_disabled=False)
        if not enabled_skills:
            return "(none)"
        lines: list[str] = []
        for skill in enabled_skills:
            parts = [f"- {skill.skill_id}: {skill.description}"]
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
            lines.append(f"- skill_id: {skill.skill_id}")
            lines.append(f"  name: {skill.display_name}")
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
        self.enabled_state[skill.skill_id] = skill.enabled
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
        for skill_id, skill in self.skills.items():
            self.enabled_state[skill_id] = bool(skill.enabled)

    def _top_suggestions(self, requested_name: str, limit: int = 3) -> list[str]:
        if not self.skills:
            return []
        options = set(self.skills.keys())
        for skill in self.skills.values():
            options.update(skill.aliases)
        key = str(requested_name or "").strip().lower()
        matches = get_close_matches(key, sorted(options), n=limit, cutoff=0.35)
        return matches[:limit]


class SkillPack:
    def __init__(
        self,
        skill_id: str,
        display_name: str,
        aliases: list[str],
        description: str,
        when_to_use: str,
        pack_dir: Path,
        skill_md_path: Path,
        command_template: Optional[List[str]],
        metadata: Any = None,
        enabled: bool = True,
    ) -> None:
        self.skill_id = skill_id
        self.display_name = display_name
        self.aliases = aliases
        self.description = description
        self.when_to_use = when_to_use
        self.pack_dir = pack_dir
        self.skill_md_path = skill_md_path
        self.command_template = command_template or []
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.enabled = enabled

    @property
    def name(self) -> str:
        """Backward-compatible display label for UI and legacy callers."""
        return self.display_name

    @property
    def definition(self) -> SkillDefinition:
        return SkillDefinition(
            skill_id=self.skill_id,
            display_name=self.display_name,
            aliases=list(self.aliases),
            source_path=str(self.skill_md_path),
        )

    def run(self, args: Any = "", session=None) -> str:
        if not self.has_runtime:
            raise SkillUnavailable(
                self.skill_id,
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
            "skill_id": self.skill_id,
            "display_name": self.display_name,
            "aliases": self.aliases,
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
    skill_id = _clean_scalar(frontmatter.get("skill_id", ""))
    description = _clean_scalar(frontmatter.get("description", ""))
    when_to_use = _extract_when_to_use(lines)
    command = _coerce_command(frontmatter.get("command"))
    aliases = _coerce_aliases(frontmatter.get("aliases"))
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
        "skill_id": skill_id,
        "aliases": aliases,
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


def _coerce_aliases(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        return [raw]
    return []


def _resolve_skill_id(parsed: dict[str, Any], display_name: str) -> str:
    explicit = str(parsed.get("skill_id", "")).strip()
    if explicit:
        return normalize_skill_key(explicit)
    return normalize_skill_key(display_name)


def _build_skill_aliases(
    *,
    display_name: str,
    skill_id: str,
    explicit_aliases: Any,
) -> list[str]:
    base_aliases = _coerce_aliases(explicit_aliases)
    generated = [
        display_name,
        skill_id,
        skill_id.replace("-", "_"),
    ]
    aliases: list[str] = []
    for item in [*base_aliases, *generated]:
        alias = str(item).strip()
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


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
