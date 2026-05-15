# RELEASE - US4 V6 Apple Edition

## 1. Scope of this document

This document describes the **planned release model** for the Apple runtime.

Important distinction:

- the starter package already has its own Node/npm release lifecycle;
- the Apple runtime release flow described here is **planned**, not implemented yet in this repo state.

## 2. Runtime release policy

The runtime release source of truth will be a SemVer tag on `main`.

- `main` may produce internal artifacts;
- versioned runtime releases come from tags;
- prereleases use `-alpha`, `-beta`, or `-rc.N`.

## 3. Planned prerelease policy

| Suffix | Audience | Tap policy |
|---|---|---|
| `-alpha` | internal or narrow external testers | no `latest` |
| `-beta` | broader external testers | no `latest` |
| `-rc.N` | release-candidate testers | optional dedicated prerelease channel |
| none | general public | promoted to `latest` |

## 4. Planned runtime artifacts

When the runtime exists, release outputs are expected to include:

- `us4-cli`
- runtime dynamic library
- checksum
- release notes

These files do not exist yet in the current repo state.

## 5. Gating by project phase

### Before runnable inference exists

- no runtime release;
- only starter/bootstrap package validation;
- planning/docs can still evolve.

### After runnable CLI baseline exists

- build gate
- `ctest` gate for native smoke and contract runners
- benchmark smoke gate
- Playwright CLI smoke with evidence

### After inference baseline exists

- correctness gate
- dedicated regression gate
- benchmark evidence

## 6. Planned release automation

A future runtime release workflow is expected, but it is not present yet.

Until that file exists, docs should not imply that `.github/workflows/release.yml` is already active.

## 7. Release manager checklist

- runtime version bumped in the runtime build system
- changelog updated
- build green
- test gates green for the current maturity phase
- release tag created from `main`
- release notes published

If the runtime release automation is not yet implemented, release steps remain manual by definition.
