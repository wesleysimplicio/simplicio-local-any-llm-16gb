# Benchmarks

Este diretorio guarda harnesses de benchmark e os contratos de observabilidade
que sustentam comparacoes honestas entre backends.

## Estado atual

- `dense_baseline.cpp` mede o scaffold denso atual e imprime metadados de
  backend, fallback, token count, tempo, `asset_format`, `asset_path`,
  `mode`, fingerprint do texto gerado, telemetria de mixed dispatch ANE/Metal e
  estado termico observado pelo runtime.
- Os casos `dense-qwen/ane-requested` e `llama-fixture/ane-requested`
  registram explicitamente o caminho ANE. Em hosts sem chip M5+ eles devem
  aparecer como fallback observavel, com `observed_backend` diferente de `ane`,
  `fell_back=true` e `mixed_dispatch_strategy=disabled`.
- Os numeros atuais ainda sao evidencia de contrato, nao prova de throughput
  real de Llama 3/4.
- `real_forward_throughput.cpp` (issue #81.12) mede tokens/s, latencia de
  decode (mean/min/max sobre 5 repeticoes) e RSS do processo sobre o forward
  REAL (pesos reais de `toy-dense-real`/`toy-llama-real`, ver #85/#105) lado a
  lado com o mesmo adapter no caminho totalmente sintetico (`asset=nullptr`),
  para a comparacao explicita real-vs-sintetico que o DoD da epic #81 pede.
- `kernel_cpu_benchmarks.cpp` mede kernels CPU/NEON de forma isolada
  (`scalar_matmul`, `neon_matmul`, `scalar_attention`, `neon_attention`) e
  explicita `requested_backend`, `observed_backend`, `backend_reason`,
  `kernel_flavor`, dimensoes, repeats, `elapsed_ms_(mean|min|max)`,
  `throughput_gops` e `output_checksum`.

## Evidencia minima para o vertical Llama

Quando o benchmark especifico de Llama chegar em `T07.6`, cada execucao deve
registrar pelo menos:

- `benchmark`
- `model`
- `asset_format`
- `requested_backend`
- `observed_backend`
- `backend_reason`
- `generated_tokens`
- `elapsed_ms`
- `text_fingerprint`
- `recommended_mode` ou `mode`
- `mixed_dispatch_strategy`
- `mixed_dispatch_metal_stages`
- `mixed_dispatch_ane_stages`
- `ane_compiled_layers`
- `ane_prediction_calls`
- `thermal_pressure_level`
- `thermal_reason`
- `thermal_downgraded`
- perfil de hardware relevante para comparar runs

Enquanto a trilha de corretude contra referencia HF nao existir, o benchmark
deve emitir um placeholder explicito (`correctness_delta=pending_external_reference`).
Quando a referencia existir, adicionar o delta dos primeiros 64 tokens no
mesmo output ou em um artefato vizinho.

## Regras

- Nao transformar numero de fixture em claim de performance.
- Nao esconder fallback.
- Nao comparar runs sem hardware profile.
- Nao marcar um backend como suportado so porque o adapter apareceu no registry.
- Em hosts nao-M5, evidencia ANE valida eh a evidencia de fallback claro; nao
  substituir por numeros sinteticos.
- Em hosts sem Apple Silicon, `Metal`/`MLX` continuam bloqueados: documente o
  bloqueio, nao simule execucao nem reporte throughput desses backends.

## Comandos uteis

```bash
./build/runtime/benchmarks/dense_baseline
./build/runtime/benchmarks/real_forward_throughput
./build/runtime/benchmarks/kernel_cpu_benchmarks
python runtime/benchmarks/repro_harness.py probe-host
python runtime/benchmarks/repro_harness.py run --template docs/benchmarks/fixtures/issue-118-126.template.json --output out/issue-118-126.plan.json --dry-run
python runtime/benchmarks/repro_harness.py validate --input out/issue-118-126.plan.json
./build/apps/us4-cli run --model-path tests/fixtures/models/llama-3.1-8b --prompt "hello" --json
./build/apps/us4-cli run --model-path tests/fixtures/models/llama-3.1-8b/toy-llama.gguf --prompt "hello" --json
```

## Harness reproduzivel para #118 / #126

Para baseline real de 16 GB, eval/perplexity e captura auditavel de
stdout/stderr sem preencher markdown na mao, use:

- `runtime/benchmarks/repro_harness.py`
- `docs/benchmarks/fixtures/issue-118-126.template.json`
- `docs/benchmarks/repro-bench-run.schema.json`
- `docs/baselines/16gb-host-baseline-prep.md`

Esse fluxo existe justamente para separar:

- benchmark de contrato/fixture;
- benchmark real em host 16 GB;
- eval de qualidade reproduzivel com checkpoint e dataset reais.

## `real_forward_throughput`: metodologia e resultado (issue #81.12)

Metodologia (principio 4 da epic #81: numeros tem que vir com contexto pra
serem honestos):

- **Hardware**: reportado pelo proprio benchmark (`platform`/`architecture`/
  `chip`/`has_neon`). Roda em qualquer host; os numeros abaixo sao de uma
  sandbox Linux x86_64 SEM NEON -- por isso os casos `neon-requested`
  observam fallback para scalar (comportamento correto e documentado, nao um
  bug: NEON so acelera de fato em ARM64/Apple Silicon real).
- **Modelos**: `toy-dense-real` (Qwen, fp32, vocab de 4 tokens) e
  `toy-llama-real` (Llama, fp32, hidden_size=4, vocab de 4 tokens) -- os
  mesmos fixtures usados pelos testes de contrato de #85/#105, NAO um
  checkpoint de producao (isso e o escopo de #81.11/#106, ainda pendente).
  Os numeros absolutos de tokens/s aqui NAO extrapolam pra um modelo real de
  bilhoes de parametros; o que e comparavel e a RAZAO entre o caminho real e
  o sintetico sob a mesma carga.
- **Quantizacao**: nenhuma -- ambos os fixtures sao fp32, pra isolar o custo
  do forward real (leitura/uso de tensor real) do custo de dequantizacao
  (que #89 ja mede separadamente).
- **Contexto**: prompt vazio (cai no `default_prompt_token` do manifest) +
  `max_tokens_per_run=64` gerados, greedy/argmax determinístico (sem
  temperatura de sampling -- `decoding=` no output documenta isso
  explicitamente).
- **Repeticoes**: 5 por caso, reportando mean/min/max de latencia pra dar um
  sinal de estabilidade (nao so um numero isolado).
- **Memoria**: RSS do processo (`/proc/self/status:VmRSS`, Linux; reporta
  `-1` em outras plataformas) antes/depois das 5 repeticoes de cada caso.

Resultado de uma rodada nesta sandbox (Linux x86_64, sem NEON):

```
benchmark=real_forward_throughput
platform=linux
architecture=x64
chip=generic-host
has_neon=false
max_tokens_per_run=64
repeats_per_case=5
decoding=greedy-argmax-deterministic-no-sampling-temperature
--
case=qwen-real-vs-synthetic/scalar
real_tokens_per_second_mean=240033
real_latency_ms_mean=0.282377
synthetic_tokens_per_second_mean=313065
synthetic_latency_ms_mean=0.208916
real_vs_synthetic_latency_ratio=1.35163
rss_delta_kib=504
--
case=llama-real-vs-synthetic/neon-gqa
real_tokens_per_second_mean=443777
real_latency_ms_mean=0.178604
synthetic_tokens_per_second_mean=400646
synthetic_latency_ms_mean=0.16539
real_vs_synthetic_latency_ratio=1.0799
rss_delta_kib=0
--
```

Leitura honesta: o caminho real e ~8-38% mais lento que o sintetico nesses
fixtures pequenos (custo de indexar um tensor real em vez de gerar um valor
deterministico inline), o que e o esperado e nao uma regressao -- o forward
sintetico e, por definicao, mais barato porque nao faz I/O nenhum sobre
dado real. RSS praticamente nao muda entre antes/depois das 5 repeticoes
(sem leak aparente nesses fixtures pequenos). Rodar em hardware ARM64 real
com NEON habilitado, e eventualmente contra um checkpoint de producao
(#81.11/#106), e o proximo passo pra numeros que generalizem alem desta
sandbox.
