# CONTRIBUTING - US4 V6 Apple Edition

## 1. Know the phase you are changing

Before editing, classify the work:

- `starter/bootstrap`
- `planning/docs`
- `runtime implementation`

This matters because validation and expectations differ.

## 2. Read in order

1. `AGENTS.md`
2. current sprint `SPRINT.md`
3. target `*.task.md`
4. relevant product and architecture docs
5. relevant ADRs

## 3. Branch naming

- `feat/<task-id>-<slug>`
- `fix/<task-id>-<slug>`
- `docs/<slug>`
- `chore/<slug>`

Examples:

- `feat/t01.2-cli-contract`
- `feat/t02.4-dense-adapter-base`
- `docs/planning-alignment`

## 4. Validation expectations

### Starter/bootstrap work

- `npm run lint`
- `npm test -- --coverage`
- `npm run test:cli`
- `npm run pack:dry`

### Planning/docs work

- placeholder sweep
- broken-link sweep
- consistency between README, specs, and AGENTS rules

### Runtime work

- `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
- `cmake --build build`
- `ctest --test-dir build --output-on-failure`
- `./build/runtime/benchmarks/dense_baseline`
- `npx playwright test`

Correctness diff and a dedicated benchmark/regression matrix stay phase-gated
until `runtime/benchmarks/correctness/` and the broader inference harness exist.

## 5. PR expectations

Include:

- linked task;
- what phase the PR belongs to;
- validation run;
- follow-up work not included.

Architecture changes should cite the matching ADR.
