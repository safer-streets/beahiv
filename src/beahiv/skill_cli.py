"""Install or remove beahiv's agent skill (`skill/SKILL.md`) under an agent's config directory.

    beahiv-skill install .claude      # -> .claude/skills/beahiv/
    beahiv-skill remove .agents       # removes .agents/skills/beahiv/

Run it from a project that depends on beahiv, so the installed skill always matches the installed
library version. Also runnable as `python -m beahiv.skill_cli`.
"""

import argparse
import sys
from importlib.resources import files
from pathlib import Path

SKILL_NAME = "beahiv"


def _skill_files() -> dict[str, bytes]:
    source = files("beahiv") / "skill"
    return {entry.name: entry.read_bytes() for entry in source.iterdir() if entry.is_file()}


def skill_dir(root: Path) -> Path:
    """Where the skill lives under an agent config directory such as `.claude` or `.agents`."""
    return root / "skills" / SKILL_NAME


def _modified(destination: Path, skill: dict[str, bytes]) -> list[Path]:
    return [
        destination / name
        for name, data in skill.items()
        if (destination / name).exists() and (destination / name).read_bytes() != data
    ]


def install(root: Path, force: bool = False) -> list[str]:
    """Copy the skill into `root/skills/beahiv/`, returning a line per file describing the outcome.

    A file that differs from the packaged one may hold local edits, so it is only replaced with
    `force`. Without it this raises FileExistsError before writing anything, so a refused install
    never leaves a half-updated skill behind.
    """
    destination = skill_dir(root)
    skill = _skill_files()
    if (modified := _modified(destination, skill)) and not force:
        raise FileExistsError(
            f"{', '.join(map(str, modified))} differs from the packaged skill -- rerun with --force to overwrite"
        )

    destination.mkdir(parents=True, exist_ok=True)
    report = []
    for name, data in skill.items():
        target = destination / name
        if not target.exists():
            status = "installed"
        elif target.read_bytes() == data:
            report.append(f"up to date  {target}")
            continue
        else:
            status = "overwrote"
        target.write_bytes(data)
        report.append(f"{status:<11} {target}")
    return report


def remove(root: Path, force: bool = False) -> list[str]:
    """Delete the skill's files from `root/skills/beahiv/`, then the directory if that empties it.

    Only files the skill ships are touched; anything else in the directory is left, and keeps the
    directory alive. Locally edited files need `force`, checked up front as for `install`.
    """
    destination = skill_dir(root)
    skill = _skill_files()
    if (modified := _modified(destination, skill)) and not force:
        raise FileExistsError(f"{', '.join(map(str, modified))} has local edits -- rerun with --force to remove")

    report = []
    for name in skill:
        target = destination / name
        if target.exists():
            target.unlink()
            report.append(f"removed     {target}")
    if destination.is_dir() and not any(destination.iterdir()):
        destination.rmdir()
        report.append(f"removed     {destination}")
    return report or [f"not installed  {destination}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beahiv-skill",
        description="Install or remove the beahiv agent skill under ROOT/skills/beahiv/.",
    )
    parser.add_argument("action", choices=("install", "remove"))
    parser.add_argument("root", type=Path, help="agent config directory, e.g. .claude, .agents or ~/.claude")
    parser.add_argument("--force", action="store_true", help="overwrite or remove locally edited files")
    args = parser.parse_args(argv)

    action = install if args.action == "install" else remove
    try:
        report = action(args.root, force=args.force)
    except FileExistsError as e:
        print(f"beahiv-skill: {e}", file=sys.stderr)
        return 1
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
