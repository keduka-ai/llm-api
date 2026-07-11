"""Invariants for the .claude/agents/ tree.

Mirrors the agent-tree hardening used in the KAIS project
(test_claude_agents_invariants.py there): every agent file carries YAML
frontmatter the Claude Code loader understands, shared rules live in one
canonical SHARED_RULES.md instead of being duplicated inline, INDEX.md stays
complete, and the Agent(...) delegation graph resolves with no self-loops.
"""

import re
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "agents"

# Documentation-only files — no frontmatter, ignored by the agent loader.
NON_AGENT_FILES = {"SHARED_RULES.md", "INDEX.md"}

# The full expected agent roster.
EXPECTED_AGENTS = {
    "orchestrator",
    "handler",
    "devops",
    "tester",
    "reviewer",
    "docs",
    "debugger",
    "security",
    "perf",
    "llm-client",
}

# Agents that must not be able to mutate files.
READ_ONLY_AGENTS = {"orchestrator", "reviewer", "debugger", "perf"}


def agent_files():
    return sorted(
        p for p in AGENTS_DIR.glob("*.md") if p.name not in NON_AGENT_FILES
    )


def parse_frontmatter(path):
    """Return (frontmatter_text, body_text) or (None, full_text)."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, flags=re.DOTALL)
    if not match:
        return None, text
    return match.group(1), match.group(2)


def frontmatter_value(frontmatter, key):
    """Extract a (possibly folded multi-line) scalar value for key."""
    match = re.search(
        rf"^{key}:\s*(.*(?:\n[ \t]+.*)*)", frontmatter, flags=re.MULTILINE
    )
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith(">"):
        value = " ".join(
            line.strip() for line in value.splitlines()[1:]
        ).strip()
    return value


class TestAgentRoster:
    def test_agents_dir_exists(self):
        assert AGENTS_DIR.is_dir()

    def test_expected_agents_all_present(self):
        present = {p.stem for p in agent_files()}
        missing = EXPECTED_AGENTS - present
        assert not missing, f"missing agent files: {sorted(missing)}"

    def test_no_unexpected_agents(self):
        present = {p.stem for p in agent_files()}
        unexpected = present - EXPECTED_AGENTS
        assert not unexpected, (
            f"unlisted agent files (add to EXPECTED_AGENTS + INDEX.md): "
            f"{sorted(unexpected)}"
        )


class TestFrontmatter:
    @pytest.mark.parametrize("path", agent_files(), ids=lambda p: p.stem)
    def test_has_frontmatter_with_required_keys(self, path):
        frontmatter, _ = parse_frontmatter(path)
        assert frontmatter is not None, f"{path.name} has no YAML frontmatter"
        for key in ("name", "description", "tools", "model", "maxTurns"):
            assert frontmatter_value(frontmatter, key), (
                f"{path.name} frontmatter missing '{key}'"
            )

    @pytest.mark.parametrize("path", agent_files(), ids=lambda p: p.stem)
    def test_name_matches_filename(self, path):
        frontmatter, _ = parse_frontmatter(path)
        assert frontmatter is not None
        assert frontmatter_value(frontmatter, "name") == path.stem

    @pytest.mark.parametrize("stem", sorted(READ_ONLY_AGENTS))
    def test_read_only_agents_disallow_writes(self, stem):
        path = AGENTS_DIR / f"{stem}.md"
        frontmatter, _ = parse_frontmatter(path)
        assert frontmatter is not None
        disallowed = frontmatter_value(frontmatter, "disallowedTools") or ""
        assert "Write" in disallowed and "Edit" in disallowed, (
            f"{stem} must disallow Write and Edit"
        )


class TestDelegationGraph:
    @pytest.mark.parametrize("path", agent_files(), ids=lambda p: p.stem)
    def test_agent_references_resolve_without_self_loops(self, path):
        frontmatter, _ = parse_frontmatter(path)
        assert frontmatter is not None
        tools = frontmatter_value(frontmatter, "tools") or ""
        match = re.search(r"Agent\(([^)]*)\)", tools)
        if not match:
            return
        referenced = {name.strip() for name in match.group(1).split(",")}
        present = {p.stem for p in agent_files()}
        unresolved = referenced - present
        assert not unresolved, (
            f"{path.stem} delegates to unknown agents: {sorted(unresolved)}"
        )
        assert path.stem not in referenced, f"{path.stem} delegates to itself"


class TestSharedRules:
    def test_shared_rules_exists_without_frontmatter(self):
        path = AGENTS_DIR / "SHARED_RULES.md"
        assert path.is_file()
        frontmatter, _ = parse_frontmatter(path)
        assert frontmatter is None, "SHARED_RULES.md must not be an agent"

    @pytest.mark.parametrize("rule", [f"R{n}" for n in range(1, 10)])
    def test_shared_rules_defines_rule(self, rule):
        text = (AGENTS_DIR / "SHARED_RULES.md").read_text(encoding="utf-8")
        assert re.search(rf"^## {rule} — ", text, flags=re.MULTILINE), (
            f"SHARED_RULES.md missing '## {rule} — ...' section"
        )

    @pytest.mark.parametrize("path", agent_files(), ids=lambda p: p.stem)
    def test_every_agent_references_shared_rules(self, path):
        _, body = parse_frontmatter(path)
        assert "SHARED_RULES.md" in body, (
            f"{path.name} must reference .claude/agents/SHARED_RULES.md"
        )

    @pytest.mark.parametrize("path", agent_files(), ids=lambda p: p.stem)
    def test_mutating_agents_carry_critic_protocol(self, path):
        frontmatter, body = parse_frontmatter(path)
        assert frontmatter is not None
        disallowed = frontmatter_value(frontmatter, "disallowedTools") or ""
        if "Write" in disallowed and "Edit" in disallowed:
            return  # read-only agents inherit R9 via the shared-rules pointer
        assert "Critic-Evaluator" in body, (
            f"{path.name} mutates files but has no Investigate-First + "
            f"Critic-Evaluator protocol section"
        )


class TestIndex:
    def test_index_exists_and_is_complete(self):
        path = AGENTS_DIR / "INDEX.md"
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        for stem in sorted(p.stem for p in agent_files()):
            assert f"`{stem}`" in text, f"INDEX.md missing row for `{stem}`"

    def test_index_has_no_stale_rows(self):
        text = (AGENTS_DIR / "INDEX.md").read_text(encoding="utf-8")
        listed = set(re.findall(r"^\| `([a-z0-9-]+)` \|", text, flags=re.MULTILINE))
        present = {p.stem for p in agent_files()}
        stale = listed - present
        assert not stale, f"INDEX.md lists agents that do not exist: {sorted(stale)}"
