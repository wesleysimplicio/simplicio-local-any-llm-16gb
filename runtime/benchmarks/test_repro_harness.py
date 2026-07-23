import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HARNESS = HERE / "repro_harness.py"
TEMPLATE = REPO / "docs/benchmarks/fixtures/issue-118-126.template.json"


class ReproHarnessValidationTest(unittest.TestCase):
    def test_repository_template_has_no_unresolved_execution_inputs(self):
        sys.path.insert(0, str(HERE))
        from repro_harness import _load_json, validate_template

        self.assertEqual(validate_template(_load_json(TEMPLATE)), [])

    def test_rejects_duplicate_ids_and_undefined_variables(self):
        sys.path.insert(0, str(HERE))
        from repro_harness import validate_template

        errors = validate_template({
            "suite_id": "bad",
            "variables": {},
            "cases": [
                {"id": "same", "command": "echo ${missing}"},
                {"id": "same", "command": "echo ok"},
            ],
        })
        self.assertIn("undefined variable 'missing'", " ".join(errors))
        self.assertIn("duplicates 'same'", " ".join(errors))

    def test_dry_run_records_template_and_command_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.json"
            result = subprocess.run(
                [sys.executable, str(HARNESS), "run", "--template", str(TEMPLATE),
                 "--output", str(output), "--dry-run"],
                cwd=REPO, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertRegex(payload["template_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(payload["cases"])
        self.assertTrue(all(len(case["command_sha256"]) == 64 for case in payload["cases"]))


if __name__ == "__main__":
    unittest.main()
