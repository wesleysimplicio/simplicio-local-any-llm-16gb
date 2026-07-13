#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_DIR = REPO_ROOT / "engine" / "c"
FIXTURE_DIR = Path(__file__).resolve().parent / "tokenizer"
SOURCE = ENGINE_DIR / "tests" / "test_tok.c"
TOKENIZER = FIXTURE_DIR / "minimal_tokenizer.json"
CASES = FIXTURE_DIR / "minimal_cases.tsv"


def compiler() -> str | None:
    for candidate in (os.environ.get("CC"), "clang", "gcc", "cc"):
        if candidate and shutil.which(candidate):
            return candidate
    return None


def main() -> int:
    cc = compiler()
    if not cc:
        print("SKIP: nenhum compilador C encontrado; contrato do tokenizer roda no runner Linux.")
        return 0

    with tempfile.TemporaryDirectory(prefix="tok-contract-") as tempdir:
        binary = Path(tempdir) / ("tok_contract.exe" if os.name == "nt" else "tok_contract")
        compile_result = subprocess.run(
            [cc, "-O2", str(SOURCE), "-o", str(binary)],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if compile_result.stdout:
            print(compile_result.stdout)
        if compile_result.stderr:
            print(compile_result.stderr, file=sys.stderr)
        if compile_result.returncode != 0:
            raise SystemExit("Falha ao compilar engine/c/tests/test_tok.c")

        cases = CASES.read_text(encoding="utf-8")
        run_result = subprocess.run(
            [str(binary), str(TOKENIZER)],
            cwd=str(REPO_ROOT),
            input=cases,
            text=True,
            capture_output=True,
            check=False,
        )
        if run_result.stdout:
            print(run_result.stdout)
        if run_result.stderr:
            print(run_result.stderr, file=sys.stderr)
        if run_result.returncode != 0:
            raise SystemExit("Contrato do tokenizer falhou")
        if "ENCODE: 4/4" not in run_result.stdout or "DECODE(round-trip): 4/4" not in run_result.stdout:
            raise SystemExit("Resumo do tokenizer nao confirmou 4/4 em encode+decode")

    print("engine tokenizer contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
