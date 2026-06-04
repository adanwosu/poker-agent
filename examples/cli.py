"""Branded CLI entrypoint.

Verbs:
    pokerkit run     [--max-hands N] [--competition-id ID] [--dry-run] [--agent path]
    pokerkit replay  [--match ID | --latest]
    pokerkit test
    pokerkit version
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

VERSION = "0.18.1-competition"


def _ensure_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _print_help() -> None:
    print("""usage: pokerkit <command> [options]

commands:
  run       play a benchmark on Arena (L4 heuristic)
  llm       play with Claude as decision maker (L5)
  test      dry-run smoke test
  version   print version

examples:
  pokerkit run --dry-run --max-hands 1           # smoke test
  pokerkit run --max-hands 50                    # 50-hand preview (~3-5 min)
  pokerkit run                                   # full 500-hand run
  pokerkit llm --dry-run --mock-llm              # test LLM path offline
  pokerkit llm --max-hands 50                    # LLM preview (uses ANTHROPIC_API_KEY)
  pokerkit llm --model haiku --max-hands 50      # cheaper test
""")


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_path()
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "version":
        print(f"arena-pokerkit {VERSION}")
        return 0

    if cmd == "run":
        import agent
        return agent.main(rest)

    if cmd == "llm":
        import llm_agent
        return llm_agent.main(rest)

    if cmd == "test":
        # Quick offline smoke test
        test_args = ["--dry-run", "--max-hands", "1"] + [a for a in rest if a not in ("test",)]
        import agent
        return agent.main(test_args)

    print(f"unknown command: {cmd}", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
