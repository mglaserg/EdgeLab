"""Small guided CLI for the common EdgeLab workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .registry import Registry
from .study import Study, study


def _ask(label: str, *, default: str | None = None, optional: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}\n> ").strip()
        if value:
            return value
        if default is not None:
            return default
        if optional:
            return ""
        print("Please enter a value.")


def _cmd_new(args: argparse.Namespace) -> int:
    print("EDGE LAB — NEW STUDY\n")
    name = _ask("Short name for the study?")
    idea = _ask("What are you investigating? (plain English is fine)")
    why = _ask(
        "Why might it happen? (press Enter if you do not know)", optional=True
    )
    if not why:
        print("\nThat's okay. EdgeLab will register it as UNLABELED rather than invent a story.\n")
    universe = _ask("What market / universe?")
    target = _ask("What outcome should the signal predict?")
    feature = _ask("Feature/version label?", default="unversioned@v1")
    is_end = _ask("Last in-sample date? (YYYY-MM-DD)")
    oos_start = _ask("First out-of-sample date? (YYYY-MM-DD)")

    s = study(
        name=name,
        idea=idea,
        why=why or None,
        universe=universe,
        target=target,
        feature_spec=feature,
        is_end=is_end,
        oos_start=oos_start,
        db=args.db,
    )
    try:
        print("\nREGISTERED\n")
        print(s.summary())
        print("\nNext: use Study.diagnose(signal, returns), then Study.test(net_pnl).")
    finally:
        s.close()
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    reg = Registry(args.db)
    try:
        s = Study.load(reg, args.hypothesis_id)
        print(s.summary())
    finally:
        reg.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edgelab",
        description="EdgeLab: register, diagnose, and test trading ideas without fooling yourself.",
    )
    p.add_argument("--db", default="edgelab.db", help="SQLite registry path")
    sub = p.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="guided registration of a new study")
    new.set_defaults(func=_cmd_new)

    show = sub.add_parser("show", help="show a registered study")
    show.add_argument("hypothesis_id")
    show.set_defaults(func=_cmd_show)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
