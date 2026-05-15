---
sprint: sprint-03
status: in_progress
start: 2026-06-11
end: 2026-06-24
owner: us4-core
---

# Sprint 03 - MLX + Metal Skeleton (Apple)

## Objetivo

Integrar MLX e Metal. Command queue + kernels matmul Metal, MLX bridge, KV em memoria unificada.

## Estado atual no repo em 2026-05-14

- `runtime/core/backend_selector.{h,cpp}` ja existe e o CLI expoe `backend`, `backend_reason` e `fallback`.
- O contrato de selecao cobre `scalar`, `neon`, `mlx`, `metal` e `ane`, com fallback automatico por hardware, mode e capability.
- A geracao atual continua no scaffold deterministico compartilhado; `runtime/mlx/` e `runtime/metal/` ainda nao entregam execucao real.
- O plano abaixo continua sendo a meta de entrega completa do sprint.

## Estado atual no repo em 2026-05-15

- `runtime/metal/command_queue.{h,cpp}` existe como skeleton cross-platform e ja registra dispatches de kernels `matmul`, `softmax` e `rmsnorm`.
- `runtime/metal/kernel_library.{h,cpp}` e `runtime/metal/kernels/*.metal` agora versionam o catalogo de kernels Metal dentro do repo.
- `runtime/metal/dense_dispatch.{h,cpp}` materializa o pipeline denso `matmul -> softmax -> rmsnorm`.
- `runtime/metal/command_queue.{h,cpp}` agora tambem expoe um perfil de init (`unavailable -> device-ready -> queue-ready`) e registra a intencao de boundary autorelease por dispatch.
- `runtime/mlx/mlx_bridge.{h,cpp}` existe como skeleton cross-platform e ja registra build/eval de um dense plan sobre allocation compartilhada.
- `runtime/mlx/dense_plan.{h,cpp}` materializa o pipeline `embedding -> attention -> projection`.
- `RuntimeContext` agora expoe `metalQueue()` e `mlxBridge()`, e `UnifiedAllocator` diferencia `cpu-only` de `unified-shared`.
- Qwen e Gemma ja declaram capability de Metal e o scaffold de geracao usa esse caminho quando a selecao permite.
- A selecao de backend agora respeita melhor os modos: `metal` so em `FULL/BALANCED_PLUS`, `mlx` em modos intermediarios, `neon` nos modos baixos, com `backend_reason` explicito (`auto-metal`, `auto-mlx`, `auto-neon`, `auto-scalar`).
- O smoke test nativo e o contract runner cobrem essa infraestrutura nova sem depender de GTest.
- Ainda faltam device real, bridge real, compilacao/execucao real dos kernels `.metal` e integracao numerica nesses backends.

## Tasks

- [x] T03.1 - Metal device init + `runtime/metal/CommandQueue` + autorelease wrapper
- [x] T03.2 - `runtime/metal/kernels/matmul.metal` (FP16/BF16) + dispatch wrapper scaffold
- [x] T03.3 - `runtime/metal/kernels/{softmax,rmsnorm}.metal` scaffold
- [x] T03.4 - MLX integration `runtime/mlx/MLXBridge` (graph build, eval, buffer share) scaffold
- [x] T03.5 - `runtime/memory/UnifiedAllocator` (unified memory CPU+GPU) scaffold
- [x] T03.6 - Qwen + Gemma: enable Metal path (dispatch flag)
- [x] T03.7 - Backend selector logic (CPU/MLX/Metal) ligado a RuntimeMode

## Test plan

- Unit: matmul Metal vs scalar (atol 1e-2 FP16); softmax Metal vs scalar; MLXBridge eval.
- Regression: Sprint 02 CPU paths intactos.
- E2E: `us4-cli run --backend metal` gera 32 tokens em <=10s no M2/M3.
- Correctness: diff Metal vs CPU <= 5e-3.

## DoD

- Metal habilitado em FULL/BALANCED_PLUS.
- Coverage >=80% em `runtime/metal` + `runtime/mlx`.
- ADR-003 Backend selection strategy.

## Riscos

- MLX API churn -> pinar versao no CMake.
