# Release v1.0 - Artifact Flow

> Origem: Sprint 12 / T12.6. Define o fluxo de release v1.0 alem do que
> `RELEASE.md` ja cobre na fase planejada.

## 1. Tag and changelog

- Tag SemVer no `main`: `v1.0.0`
- `CHANGELOG.md` na raiz, com secoes `Added`, `Changed`, `Fixed`, `Removed`
- Conventional commits agregam automaticamente no body do release notes

## 2. Signed binary

- Universal arm64 binary `us4-cli` assinado com Apple Developer ID
- Checksum SHA-256 publicado lado a lado
- Empacotamento: `tar.gz` + Homebrew formula em `wesleysimplicio/us4` tap

## 3. Release workflow dependencies

- `.github/workflows/release.yml` (planejado) dispara em `git tag v*`
- Steps:
  1. Build no runner `macos-14`
  2. `ctest --test-dir build` verde
  3. `runtime/benchmarks/dense_baseline` smoke
  4. `runtime/benchmarks/matrix_runner` smoke
  5. Code signing + notarization (`xcrun notarytool`)
  6. Upload assets via `gh release create`

## 4. GA validation

- Coverage total >= 80% (linha + branch)
- Correctness diff dentro de tolerancia em todos os adapters
- Bench matrix verde para os 7 RAM tiers x 9 adapters
- Playwright smoke verde com evidencia anexa
- Demo gravado e linkado no release notes

## 5. Manual fallback

Enquanto `release.yml` nao existir:

```bash
cmake --build build --config Release
ctest --test-dir build
./build/runtime/benchmarks/dense_baseline > evidence/dense_baseline.txt
./build/runtime/benchmarks/matrix_runner > evidence/matrix_runner.txt
shasum -a 256 build/apps/us4-cli > evidence/us4-cli.sha256
gh release create v1.0.0 build/apps/us4-cli evidence/*
```

## 6. Referencias

- `.specs/workflow/RELEASE.md`
- `.specs/workflow/MIGRATION.md`
- `.specs/sprints/sprint-12/T12.6-release-v1.task.md`
