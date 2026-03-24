from __future__ import annotations

from pathlib import Path

import pytest

from app.skill_manager import (
    SkillAmbiguousError,
    SkillManager,
    SkillNotFoundError,
    SkillRegistryError,
)


def _write_skill(pack_dir: Path, *, frontmatter: str, body: str = "# Skill\n") -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    skill_md = pack_dir / "SKILL.md"
    skill_md.write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")


def _new_manager(tmp_path: Path) -> SkillManager:
    packs_dir = tmp_path / "skills"
    state_file = tmp_path / "skills_state.json"
    return SkillManager(packs_dir=str(packs_dir), state_file=str(state_file))


def test_email_slug_resolves_display_name(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "skills" / "email-mail-master-1.0.0",
        frontmatter=(
            "name: email-mail-master 万能邮箱助手\n"
            "description: 邮件技能"
        ),
    )
    manager = _new_manager(tmp_path)
    skill, matched_by = manager.resolve_skill("email-mail-master")
    assert skill.display_name == "email-mail-master 万能邮箱助手"
    assert skill.skill_id == "email-mail-master"
    assert matched_by in {"exact_id", "exact_alias", "normalized_alias"}


def test_hyphen_underscore_case_variants_resolve(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "skills" / "email-mail-master-1.0.0",
        frontmatter=(
            "name: email-mail-master 万能邮箱助手\n"
            "description: 邮件技能"
        ),
    )
    manager = _new_manager(tmp_path)
    skill, matched_by = manager.resolve_skill("EMAIL_MAIL_MASTER")
    assert skill.skill_id == "email-mail-master"
    assert matched_by == "normalized_alias"


def test_not_found_raises_with_suggestions(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "skills" / "email-mail-master-1.0.0",
        frontmatter="name: email-mail-master 万能邮箱助手\ndescription: 邮件技能",
    )
    manager = _new_manager(tmp_path)
    with pytest.raises(SkillNotFoundError) as exc_info:
        manager.resolve_skill("email-mail-mastr")
    err = exc_info.value
    assert err.requested_name == "email-mail-mastr"
    assert "email-mail-master" in err.candidates


def test_ambiguous_raises_candidates(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "skills" / "skill-a",
        frontmatter=(
            "name: alpha-mail\n"
            "description: A\n"
            "aliases: [\"mail-check\"]"
        ),
    )
    _write_skill(
        tmp_path / "skills" / "skill-b",
        frontmatter=(
            "name: beta-mail\n"
            "description: B\n"
            "aliases: [\"mail-check\"]"
        ),
    )
    manager = _new_manager(tmp_path)
    with pytest.raises(SkillAmbiguousError) as exc_info:
        manager.resolve_skill("mail-check")
    assert sorted(exc_info.value.candidates) == ["alpha-mail", "beta-mail"]


def test_duplicate_canonical_id_fails_registration(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "skills" / "first",
        frontmatter="name: email_master\ndescription: First",
    )
    _write_skill(
        tmp_path / "skills" / "second",
        frontmatter="name: email-master\ndescription: Second",
    )
    with pytest.raises(SkillRegistryError):
        _new_manager(tmp_path)


def test_no_silent_failure_on_execute_missing_skill(tmp_path: Path) -> None:
    _write_skill(
        tmp_path / "skills" / "email-mail-master-1.0.0",
        frontmatter="name: email-mail-master 万能邮箱助手\ndescription: 邮件技能",
    )
    manager = _new_manager(tmp_path)
    with pytest.raises(SkillNotFoundError):
        manager.execute("email-mail-master-typo", "")
