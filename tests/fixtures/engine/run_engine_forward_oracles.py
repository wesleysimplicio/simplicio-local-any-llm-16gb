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
FIXTURE_ROOT = Path(__file__).resolve().parent / "colibri"
GLM_FIXTURE = FIXTURE_ROOT / "glm_tiny"
DEEPSEEK_FIXTURE = FIXTURE_ROOT / "deepseek_tiny"
REF_GLM = FIXTURE_ROOT / "ref_glm.json"
REF_DEEPSEEK = FIXTURE_ROOT / "ref_deepseek.json"
GLM_BIN = ENGINE_DIR / ("glm.exe" if os.name == "nt" else "glm")


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(f"==> {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return completed


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def build_engine() -> None:
    make = shutil.which("make")
    if not make:
        print("SKIP: 'make' indisponivel neste host; suite colibri depende de runner Linux/macOS.")
        raise SystemExit(0)
    result = run([make, "glm"], cwd=ENGINE_DIR)
    require(result.returncode == 0 and GLM_BIN.exists(), "Falha ao compilar engine/c/glm")


def run_oracle_case(name: str, snap: Path, ref: Path) -> None:
    tf_env = dict(os.environ, SNAP=str(snap), REF=str(ref), TF="1")
    tf = run([str(GLM_BIN), "64", "16", "16"], cwd=ENGINE_DIR, env=tf_env)
    require(tf.returncode == 0, f"{name}: teacher-forcing falhou")
    require("PREFILL (teacher-forcing) C vs oracolo: 32/32" in tf.stdout, f"{name}: prefill nao bateu 32/32")
    require("[ORACLE] mismatch" not in tf.stderr, f"{name}: mismatch de teacher-forcing")

    greedy_env = dict(os.environ, SNAP=str(snap), REF=str(ref))
    greedy = run([str(GLM_BIN), "64", "16", "16"], cwd=ENGINE_DIR, env=greedy_env)
    require(greedy.returncode == 0, f"{name}: greedy falhou")
    require("Token coincidenti: 20/20" in greedy.stdout, f"{name}: greedy nao bateu 20/20")


def main() -> int:
    require(GLM_FIXTURE.exists(), f"Fixture ausente: {GLM_FIXTURE}")
    require(DEEPSEEK_FIXTURE.exists(), f"Fixture ausente: {DEEPSEEK_FIXTURE}")
    require(REF_GLM.exists(), f"Ref ausente: {REF_GLM}")
    require(REF_DEEPSEEK.exists(), f"Ref ausente: {REF_DEEPSEEK}")

    build_engine()
    run_oracle_case("glm_tiny", GLM_FIXTURE, REF_GLM)
    run_oracle_case("deepseek_tiny", DEEPSEEK_FIXTURE, REF_DEEPSEEK)

    with tempfile.TemporaryDirectory(prefix="colibri-i4-") as tempdir:
      outdir = Path(tempdir) / "glm_tiny_i4"
      convert = run(
          [
              sys.executable,
              str(ENGINE_DIR / "tools" / "convert_fp8_to_int4.py"),
              "--indir",
              str(GLM_FIXTURE),
              "--outdir",
              str(outdir),
              "--ebits",
              "4",
              "--io-bits",
              "8",
          ],
          cwd=REPO_ROOT,
      )
      require(convert.returncode == 0, "Conversao int4 do fixture glm_tiny falhou")
      require(any(outdir.glob("out-*.safetensors")), "Conversao int4 nao produziu shards")

    print("engine forward oracle suite: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
