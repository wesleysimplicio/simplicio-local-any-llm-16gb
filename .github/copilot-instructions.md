# copilot-instructions.md

> Espelho slim de [../AGENTS.md](../AGENTS.md) pro GitHub Copilot (Chat + Workspace + CLI). Quando GitHub Copilot abre este repositório, lê este arquivo e segue as mesmas regras de [`AGENTS.md`](../AGENTS.md). Para detalhes completos: ler `AGENTS.md`.
>
> Quando mudar `AGENTS.md`, atualize aqui também (ou trate `AGENTS.md` como única fonte da verdade e mantenha este arquivo como pointer).

---

## Projeto

- **Nome**: US4 V6 Apple Edition (`us4-v6-simplicio-apple`).
- **Domínio**: Universal State Runtime — inferência local de LLMs em Apple Silicon (M1..M5+).
- **Time**: us4-core.

## Stack

C++17/20 + CMake + MLX + Metal + NEON (Accelerate) + ANE (M5+) + GoogleTest + Playwright (CLI E2E) + Ralph Loop. CI em `macos-14`. Distribuição: tarball assinado + Homebrew tap.

## Comandos importantes

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
clang-format --dry-run --Werror $(git ls-files '*.cpp' '*.h' '*.mm')
clang-tidy -p build $(git ls-files 'runtime/**/*.cpp')
ctest --test-dir build --output-on-failure
npx playwright test
```

## Workflow OBRIGATÓRIO

`ler task -> planejar -> editar -> format -> lint -> unit -> e2e (trace+screenshot+video) -> regression + correctness -> fix loop -> conventional commit (inglês) -> PR`.

## Definition of Done

- Build verde (`cmake --build build` sem warning novo)
- Unit verde (`ctest --test-dir build`)
- Lint verde (`clang-tidy` + `clang-format --dry-run`)
- E2E Playwright verde com **evidência salva** em TODA task que toca CLI/UX (`playwright-report/` + `test-results/<spec>/trace.zip` + screenshots + video). Hard rule.
- Coverage do diff >= 80% (`llvm-cov`)
- Regression suite verde
- Correctness diff dentro da tolerância (`runtime/benchmarks/correctness/`)
- AC marcados
- PR template preenchido
- Conventional commit
- ADR se decisão arquitetural
- Changelog se release-relevant
- Sem `std::cout` / `printf` / `NSLog` de debug deixado pra trás
- Sem TODO órfão

## Proibido

- Pular testes
- Mockar pra esconder falha
- Commit vermelho
- Ignorar ADR
- Adicionar dependência sem perguntar
- Editar arquivo não lido
- Refactor escondido em PR de feature
- Force push em `main`
- Commitar segredo
- Reformatar arquivo inteiro num PR pequeno
- Quebrar correctness sem ADR

## Padrões

Detalhes em [`../.specs/architecture/PATTERNS.md`](../.specs/architecture/PATTERNS.md). Resumo: `PascalCase` classes / `camelCase` métodos / `kSnakeCase` constantes / `snake_case` arquivos. `#pragma once`. `std::unique_ptr` default. `Tensor` move-only. `std::expected<T,Error>` em ABI. No global mutable state fora de `RuntimeContext`.

## Skills disponíveis

Em `../.skills/`:

- **caveman** (default, modo terse)
- **ralph-loop** (loop autônomo, obrigatório em task com AC mensurável)
- **everything-claude-code** (~60 agents + ~221 skills, paralelo)
- **playwright-e2e** (sob demanda)
- **conventional-commits** (sob demanda)

## Custom agents

Em `../.agents/`:

- **ralph-loop.agent.md** — Ralph Loop (mapeia `/ralph-loop "<prompt>"`, `/goal`, `--autopilot`, "Autopilot", Background Agent).
- **tdd.agent.md** — TDD red-green-refactor com GoogleTest.
- **reviewer.agent.md** — Code Reviewer C++ read-only.
- **architect.agent.md** — Architect (ADRs, `PATTERNS.md`).

## Idioma

- Respostas/docs: pt-BR.
- Código (vars/funções/classes): inglês.
- Commits: inglês.
- Sem emoji em código. Sem resumo no final. Sem estimativa de tempo.

## Contexto

- Backlog: `../.specs/sprints/BACKLOG.md`.
- Sprint atual: `../.specs/sprints/sprint-XX/SPRINT.md`.
- Padrões: `../.specs/architecture/PATTERNS.md`.
- Decisões: `../.specs/architecture/ADR-*.md`.

<!-- codex-long-running-agent-overlay:start -->
## Universal Long-Running Agent Overlay

This section complements the repository-specific guidance already in this file. If anything here conflicts with the repo-specific rules above, the repo-specific rules win.

- `PRD.md` is the task source of truth for long-running sessions.
- `PROGRESS.md` is the persistent checkpoint log.
- `GOAL_RESULT.md` is the final execution report.
- Before coding, read this file, `PRD.md`, `PROGRESS.md` when it exists, `README.md`, project manifests, tests, and the relevant source folders.
- Work in small checkpoints, run the smallest relevant validation after each meaningful change, update `PROGRESS.md`, and continue until complete or genuinely blocked.
- Stop only when the requested work is complete, validation is documented, and `GOAL_RESULT.md` reflects the outcome.
- Do not rewrite unrelated architecture, fake successful validation, expose secrets, or push without explicit operator instruction for the active session.
<!-- codex-long-running-agent-overlay:end -->
