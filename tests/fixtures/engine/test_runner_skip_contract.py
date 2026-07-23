from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


FIXTURE_DIR = Path(__file__).resolve().parent


def load_runner(name: str):
    path = FIXTURE_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerSkipContractTest(unittest.TestCase):
    def test_forward_runner_fails_without_make(self) -> None:
        runner = load_runner("run_engine_forward_oracles.py")
        with mock.patch.object(runner.shutil, "which", return_value=None):
            with self.assertRaisesRegex(SystemExit, "'make' is required"):
                runner.build_engine()

    def test_tokenizer_runner_fails_without_compiler(self) -> None:
        runner = load_runner("run_engine_tokenizer_contract.py")
        with (
            mock.patch.dict(runner.os.environ, {"CC": "missing-compiler"}, clear=False),
            mock.patch.object(runner.shutil, "which", return_value=None),
            mock.patch.object(runner.subprocess, "run") as compiler,
        ):
            with self.assertRaisesRegex(SystemExit, "compiler is required"):
                runner.main()
        compiler.assert_not_called()

    def test_only_missing_checkpoint_is_optional(self) -> None:
        runner = load_runner("run_engine_forward_oracles.py")
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            self.assertFalse(
                runner.optional_checkpoint("missing", root / "snap", root / "ref.json")
            )

            snap = root / "snap"
            snap.mkdir()
            (snap / "config.json").write_text("{}", encoding="utf-8")
            (snap / "model.safetensors").write_bytes(b"fixture")
            ref = root / "ref.json"
            ref.write_text("{}", encoding="utf-8")
            self.assertTrue(runner.optional_checkpoint("present", snap, ref))


if __name__ == "__main__":
    unittest.main()
