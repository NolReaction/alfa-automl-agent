"""Run the environment check and local tests without a real API request."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> int:
    checks = (
        ("1/2: Checking Python and installed packages", ["solution.py", "--check"]),
        ("2/2: Running local tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    )
    for description, arguments in checks:
        print(description, flush=True)
        try:
            result = subprocess.run([sys.executable, *arguments], cwd=ROOT, check=False)
        except OSError as error:
            print(f"Could not start the check: {error}", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print("Check failed. Fix the reported error and run again.", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
    print("All development checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
