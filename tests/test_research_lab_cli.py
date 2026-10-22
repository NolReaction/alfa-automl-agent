"""Machine JSON transport under redirected Windows code pages; no model fits."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def redirected_cli(*, result=None, error=None):
    mocked = (f"side_effect=ValueError({ascii(error)})" if error is not None
              else f"return_value={ascii(result)}")
    code = (
        "import sys\nfrom pathlib import Path\nfrom unittest.mock import patch\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "sys.stdout.reconfigure(encoding='cp1252', errors='strict')\n"
        "sys.stderr.reconfigure(encoding='cp1252', errors='strict')\n"
        "assert sys.flags.isolated == 1\n"
        "from research_lab.__main__ import main\n"
        f"with patch('research_lab.runner.run_plan', {mocked}) as run:\n"
        " code = main(['run', '--output', 'unused'])\n"
        " run.assert_called_once_with(Path('unused'), resume=False)\n"
        "raise SystemExit(code)\n")
    return subprocess.run([sys.executable, "-I", "-c", code],
                          capture_output=True, timeout=15)


class ResearchLabCliTests(unittest.TestCase):
    def test_success_json_survives_redirected_cp1252_without_losing_unicode(self):
        payload = {"status": "completed", "label": "Проверка модели · готово"}
        process = redirected_cli(result=payload)
        self.assertEqual(process.returncode, 0, process.stderr.decode("ascii", errors="replace"))
        self.assertEqual(json.loads(process.stdout.decode("ascii")), payload)
        self.assertEqual(process.stderr, b"")

    def test_error_json_survives_redirected_cp1252_without_losing_unicode(self):
        message = "Не найден путь: C:\\Модели\\проверка"
        process = redirected_cli(error=message)
        self.assertEqual(process.returncode, 2, process.stderr.decode("ascii", errors="replace"))
        self.assertEqual(process.stdout, b"")
        self.assertEqual(json.loads(process.stderr.decode("ascii")), {
            "status": "blocked", "error_type": "ValueError", "reason": message})


if __name__ == "__main__":
    unittest.main()
