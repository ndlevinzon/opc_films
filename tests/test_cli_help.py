"""Every command-line script can print --help.

Argparse interpolates ``%`` in help strings. An unescaped percent used to
raise ``ValueError`` before the help was shown.
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _scripts():
    found = []
    for folder in ("analysis", "MD", "src"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name == "__init__.py" or "test" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "ArgumentParser" in text and "__main__" in text:
                found.append(path)
    return found


class HelpTest(unittest.TestCase):
    def test_help_exits_zero(self):
        failures = []
        for path in _scripts():
            completed = subprocess.run(
                [sys.executable, str(path), "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            combined = completed.stdout + completed.stderr
            if "ModuleNotFoundError" in combined or "ImportError" in combined:
                continue
            if completed.returncode != 0:
                failures.append(f"{path.relative_to(ROOT)}: {combined[-400:]}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
