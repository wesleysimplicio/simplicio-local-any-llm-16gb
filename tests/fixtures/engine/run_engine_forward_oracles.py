#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_DIR = REPO_ROOT / "engine" / "c"
FIXTURE_ROOT = Path(__file__).resolve().parent / "colibri"
GLM_FIXTURE = FIXTURE_ROOT / "glm_tiny"
DEEPSEEK_FIXTURE = FIXTURE_ROOT / "deepseek_tiny"
REF_GLM = FIXTURE_ROOT / "ref_glm.json"
REF_DEEPSEEK = FIXTURE_ROOT / "ref_deepseek.json"
KIMI_FIXTURE = FIXTURE_ROOT / "kimi_tiny"
REF_KIMI = FIXTURE_ROOT / "ref_kimi.json"
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
        raise SystemExit("ERROR: 'make' is required to run the engine forward oracle suite.")
    result = run([make, "glm"], cwd=ENGINE_DIR)
    require(result.returncode == 0 and GLM_BIN.exists(), "Falha ao compilar engine/c/glm")


def generated_tokens(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("Motore C GLM"):
            return line.partition(":")[2].strip()
    raise SystemExit("Saida do motor nao contem a sequencia gerada")


def run_generation(name: str, snap: Path, ref: Path, **overrides: str) -> str:
    env = dict(os.environ, SNAP=str(snap), REF=str(ref), **overrides)
    result = run([str(GLM_BIN), "64", "16", "16"], cwd=ENGINE_DIR, env=env)
    require(result.returncode == 0, f"{name}: geracao falhou")
    require("Token coincidenti: 20/20" in result.stdout, f"{name}: greedy nao bateu 20/20")
    return generated_tokens(result.stdout)


def run_oracle_case(name: str, snap: Path, ref: Path) -> None:
    tf_env = dict(os.environ, SNAP=str(snap), REF=str(ref), TF="1")
    tf = run([str(GLM_BIN), "64", "16", "16"], cwd=ENGINE_DIR, env=tf_env)
    require(tf.returncode == 0, f"{name}: teacher-forcing falhou")
    require("PREFILL (teacher-forcing) C vs oracolo: 32/32" in tf.stdout, f"{name}: prefill nao bateu 32/32")
    require("[ORACLE] mismatch" not in tf.stderr, f"{name}: mismatch de teacher-forcing")

    baseline = run_generation(name, snap, ref, DRAFT="0", ABSORB="0", DSA="0")
    require(
        run_generation(name, snap, ref, DRAFT="0", ABSORB="1", DSA="0") == baseline,
        f"{name}: absorption alterou a saida greedy",
    )
    require(
        run_generation(
            name, snap, ref, DRAFT="0", ABSORB="0", DSA="1",
            DSA_FORCE="1", DSA_TOPK="4096",
        ) == baseline,
        f"{name}: DSA-full divergiu da atencao densa",
    )
    require(
        run_generation(name, snap, ref, DRAFT="4", ABSORB="0", DSA="0") == baseline,
        f"{name}: speculative decoding nao foi lossless",
    )


def optional_checkpoint(name: str, snap: Path, ref: Path) -> bool:
    present = snap.is_dir() and (snap / "config.json").is_file() \
        and (snap / "model.safetensors").is_file() and ref.is_file()
    if not present:
        try:
            snap_label = snap.relative_to(REPO_ROOT)
            ref_label = ref.relative_to(REPO_ROOT)
        except ValueError:
            snap_label, ref_label = snap, ref
        print(
            f"SKIP checkpoint {name}: fixture tiny/ref ausente "
            f"({snap_label}, {ref_label})"
        )
    return present


def main() -> int:
    require(GLM_FIXTURE.exists(), f"Fixture ausente: {GLM_FIXTURE}")
    require(DEEPSEEK_FIXTURE.exists(), f"Fixture ausente: {DEEPSEEK_FIXTURE}")
    require(REF_GLM.exists(), f"Ref ausente: {REF_GLM}")
    require(REF_DEEPSEEK.exists(), f"Ref ausente: {REF_DEEPSEEK}")

    build_engine()
    run_oracle_case("glm_tiny", GLM_FIXTURE, REF_GLM)
    run_oracle_case("deepseek_tiny", DEEPSEEK_FIXTURE, REF_DEEPSEEK)
    if optional_checkpoint("kimi_tiny", KIMI_FIXTURE, REF_KIMI):
        run_oracle_case("kimi_tiny", KIMI_FIXTURE, REF_KIMI)

    print("engine forward oracle suite: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
