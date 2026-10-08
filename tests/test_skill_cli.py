"""`beahiv-skill install|remove ROOT` and the agent skill it ships."""

import re
import tempfile
from importlib.resources import files
from pathlib import Path

import pytest

import beahiv
from beahiv.skill_cli import install, main, remove, skill_dir

_PACKAGED = (files("beahiv") / "skill" / "SKILL.md").read_bytes()

# Deprecated spellings the skill deliberately steers away from rather than documenting.
_DEPRECATED = {"latlon_to_cell"}


def test_skill_dir_is_under_root_skills():
    assert skill_dir(Path(".claude")) == Path(".claude/skills/beahiv")
    assert skill_dir(Path(".agents")) == Path(".agents/skills/beahiv")


def test_install_copies_the_packaged_skill():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / ".claude"
        assert main(["install", str(root)]) == 0
        assert (root / "skills" / "beahiv" / "SKILL.md").read_bytes() == _PACKAGED


def test_reinstall_is_a_no_op():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install(root)
        assert install(root) == [f"up to date  {skill_dir(root) / 'SKILL.md'}"]


def test_install_keeps_local_edits_without_force():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install(root)
        (skill_dir(root) / "SKILL.md").write_text("edited locally")

        assert main(["install", str(root)]) == 1
        assert (skill_dir(root) / "SKILL.md").read_text() == "edited locally"

        assert main(["install", str(root), "--force"]) == 0
        assert (skill_dir(root) / "SKILL.md").read_bytes() == _PACKAGED


def test_remove_deletes_the_skill_directory_but_not_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / ".agents"
        install(root)
        assert main(["remove", str(root)]) == 0
        assert not skill_dir(root).exists()
        assert (root / "skills").is_dir()


def test_remove_leaves_files_it_did_not_install():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install(root)
        (skill_dir(root) / "notes.md").write_text("mine")

        remove(root)
        assert not (skill_dir(root) / "SKILL.md").exists()
        assert (skill_dir(root) / "notes.md").read_text() == "mine"


def test_remove_keeps_local_edits_without_force():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install(root)
        (skill_dir(root) / "SKILL.md").write_text("edited locally")

        assert main(["remove", str(root)]) == 1
        assert (skill_dir(root) / "SKILL.md").exists()
        with pytest.raises(FileExistsError, match="--force"):
            remove(root)

        assert main(["remove", str(root), "--force"]) == 0
        assert not skill_dir(root).exists()


def test_remove_when_not_installed_is_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        assert remove(Path(tmp)) == [f"not installed  {skill_dir(Path(tmp))}"]


@pytest.mark.parametrize("argv", [[], ["install"], ["uninstall", ".claude"]])
def test_requires_an_action_and_a_root(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_skill_has_frontmatter_naming_it():
    text = _PACKAGED.decode()
    frontmatter = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert frontmatter is not None
    assert re.search(r"^name: beahiv$", frontmatter.group(1), re.MULTILINE)
    assert re.search(r"^description: ", frontmatter.group(1), re.MULTILINE)


def test_skill_mentions_every_public_callable():
    """Keeps the skill in step with `__all__`: a new public function fails here until documented."""
    text = _PACKAGED.decode()
    public = {name for name in beahiv.__all__ if callable(getattr(beahiv, name))} - _DEPRECATED
    missing = sorted(name for name in public if not re.search(rf"\b{name}\b", text))
    assert missing == []
