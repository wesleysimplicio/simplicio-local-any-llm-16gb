"""Build-graph regressions for the vendored C engine (issue #122)."""

import subprocess
import unittest
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parents[1]


class MakefileGraphTest(unittest.TestCase):
    def test_glm_target_has_no_self_dependency_warning(self):
        result = subprocess.run(
            ["make", "-n", "glm"],
            cwd=ENGINE_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout + result.stderr
        self.assertNotIn("Circular glm <- glm dependency dropped", output)


if __name__ == "__main__":
    unittest.main()

