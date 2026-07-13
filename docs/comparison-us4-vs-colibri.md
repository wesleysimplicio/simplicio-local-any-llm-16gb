# Comparação técnica: US4 V6 (este repo) vs colibri

> Data: 2026-07-13. Fontes: auditoria de código de ambos os repositórios nesta sessão
> (US4 em `docs/assessment-llm-tier1-16gb.md`; colibri clonado de
> `wesleysimplicio/colibri`, fork de `JustVugg/colibri`, commit `8f5f3e3`, Apache-2.0).

## Veredito de uma linha

O **colibri é hoje um motor de inferência real** que executa um MoE de fronteira
(GLM-5.2 744B int4) por streaming de experts do disco com ~10 GB residentes; o **US4 é
hoje uma plataforma de engenharia** (arquitetura multi-backend, 222 testes unitários +
29 E2E, specs/ADRs, CI/DoD) cujo motor ainda opera em escala de fixture. Em "resultado
de inferência", o colibri está na frente; em processo, verificação e produto, o US4
está na frente. O colibri é exatamente a barra que a épica #81 declarou querer bater.

## Comparação por subsistema

| Subsistema | US4 V6 | colibri |
|---|---|---|
| Forward transformer | Toy: embedding + atenção single-head + projeção; `hiddenSize=8` fixo; sem laço de camadas | **Real e fiel**: laço completo, MLA (q/kv-LoRA, weight absorption), RoPE parcial, RMSNorm, SwiGLU, router sigmoid noaux_tc, shared expert (`glm.c:1366,1006,1259`) |
| Escala de modelo | Fixtures de 426 bytes, vocab 4–16 | **GLM-5.2 744B MoE int4** (~370 GB em disco) + OLMoE |
| Loader | safetensors só F32, um arquivo, 2 tensores, sem mmap | safetensors F32/F16/BF16 **multi-shard** (até 512), container quantizado próprio `.qs`, streaming `pread`+`fadvise`+`O_DIRECT` (`st.h`) |
| GGUF | Não | Não (empate — oportunidade para ambos) |
| Tokenizer | BPE real mas só whitespace pre-tok | **BPE byte-level fiel** (regex cl100k como FSM, special tokens, chat template GLM) (`tok.h`) |
| Quantização | int4/int8 round-trip; não ligada ao forward real | **int8/int4/int2 no caminho quente**, dot inteiro AVX2/AVX-512-VNNI/**NEON-SDOT**, conversor FP8→int4 bit-idêntico (`glm.c:320-506`) |
| MoE / expert paging | Contábil (LFU de strings, limite em contagem) | **Real**: LRU por camada + hot-store fixado + learning cache persistente + auto-budget por `MemAvailable`, pread coalescente ~19 MB/expert, readahead assíncrono, batch-union (`glm.c:881,1211,2350`) |
| KV cache | Real em escala toy; spill SSD | **MLA comprimido** (576 floats/token, 57× menor), persistência crash-safe entre reinícios, multi-slot (`glm.c:1393,1897`) |
| Speculative decoding | Real em escala toy | **MTP nativo lossless** (cabeça do próprio modelo) + rejection sampling sob sampling + n-gram fallback + guarda adaptativa (`glm.c:1593`) |
| Sparse attention | Não | **DSA lightning indexer** top-2048, validado exato (`glm.c:1029`) |
| Sampling | Greedy/argmax | Temperature + top-p reais (`glm.c:1542`) |
| CPU SIMD | NEON real (matmul/atenção/dequant), inerte em x86 | AVX2 + **AVX-512 VNNI** + NEON-SDOT + escalar |
| GPU | Metal/MLX/ANE **sintéticos** (contadores) | Sem Metal; **CUDA experimental real** (correctness-first) |
| Servidor | HTTP nativo mínimo, agora com SSE/CORS para `/v1/chat/completions`, ainda sem sampling params | **OpenAI-compatible completo**: SSE, fila FIFO com 429, auth, usage, rejeição explícita do não-suportado (`openai_server.py`) |
| UI | Web UI React/Vite (`apps/web-chat`) + wrapper `bin/us4-cli.js chat` | Web UI React/Vite (cliente OpenAI puro) |
| Diagnóstico | `--probe` sintético | `coli plan` + `coli doctor` (read-only, JSON versionado) |
| Testes | **222 unit + 29 E2E Playwright, tudo verde** | 3 testes C + testes Python do server; **forward e tokenizer fora da suíte automatizada** |
| CI | Existe (desligado por custo) com gate de DoD | **Ausente** (só templates) |
| Correção provada | Oráculo externo Python por subsistema (padrão #82), toy-scale | Token-exact TF 32/32 vs `transformers` — mas **só contra oráculo tiny aleatório**, manual |
| Benchmarks | Placeholder declarado | Números detalhados mas **auto/comunidade-reportados, não reproduzíveis no repo**; custo de precisão do int4 nunca medido (admitido) |
| Números medidos | — | 0.07–0.11 tok/s (24 GB RAM, x86), **1.06 tok/s (Apple M5 Max, 128 GB)**, 0.37–0.40 (Ryzen 128 GB, pin aprendido) |
| Histórico/processo | Sprints, ADRs, conventional commits, revisão adversarial | 1 commit squashed, sem rastro auditável |
| Robustez | Parsers com revisão de segurança (#85), fuzzing planejado (#108) | Buffers de pilha fixos (`sc[8192]`, `row[8192]`) não cobertos pela validação de config |
| Escopo de arquitetura | 6 famílias de adapter declaradas (nenhuma real) | **Travado em `n_group=1`** (GLM-5.2/DeepSeek-V3-like); DeepSeek-R1 real (n_group=8) não roda |

## Onde cada um ganha

**colibri ganha em**: tudo que é motor — forward real de fronteira, quantização aplicada,
streaming de experts, KV comprimido, speculative lossless, tokenizer fiel, server
OpenAI/SSE, e um resultado demonstrado (744B respondendo em máquina de 24 GB).

**US4 ganha em**: verificação e processo (suíte de testes ampla e automatizada, E2E
Playwright sobre o binário real, CI/DoD prontos para religar, ADRs, revisão adversarial
com bugs reais encontrados antes de merge), telemetria/probe estruturados, e a ambição
Apple-first (Metal/MLX/ANE) que o colibri não endereça.

## Como superar o colibri (aberturas concretas)

1. **Prova de correção em CI** — o colibri não tem nenhuma; nós temos o padrão pronto
   (oráculo tiny por família de modelo, rodando em Actions). Superação barata e visível.
2. **Suporte multi-família real** — colibri trava em `n_group=1`. Implementar
   group-limited routing (DeepSeek-R1/V3, `n_group=8`) e as dims do Kimi K2 entrega o
   "any tier-1" que ele não tem.
3. **Perfil 16 GB provado** — o menor datapoint dele é 24 GB (RSS 14.1 GB). Um perfil
   dedicado que garanta RSS ≤ 13 GB sem swap em máquina de 16 GB é um resultado que ele
   não demonstrou.
4. **Metal no tronco denso** — ele é CPU-only no Mac (NEON). O caminho quente warm é
   matmul-bound (datapoint 9950X: 57% matmul com disco rápido); GPU Metal + unified
   memory ataca exatamente isso. O melhor número dele em Mac (1.06 tok/s, M5 Max) é a
   marca a bater.
5. **Benchmarks e qualidade reproduzíveis** — harness versionado (fixture 313M dele é
   sintética e rotulada; podemos medir perplexity/eval real do int4, que ele admite
   nunca ter medido).
6. **Robustez** — fuzzing/sanitizers (#108) + eliminar tetos de buffer fixos.

O plano de execução está em `docs/plan-simplicio-local-any-llm-16gb.md`.
