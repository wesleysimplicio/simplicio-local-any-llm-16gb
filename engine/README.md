# engine/ — colibri inference engine (vendored)

Este diretório contém o motor de inferência C do projeto
[colibri](https://github.com/JustVugg/colibri) (Apache-2.0), vendorizado
para dentro deste repositório. Veja
[ADR-010](../.specs/architecture/ADR-010-adopt-colibri-engine.md) para a
decisão e o racional completos.

## Origem

- Upstream original: https://github.com/JustVugg/colibri
- Fork vendorizado: https://github.com/wesleysimplicio/colibri
- Commit vendorizado: `8f5f3e3a2b7bef53daa1cc8608df6d91fc9a54eb` (também
  registrado em `NOTICE` na raiz deste repositório).
- Licença: Apache License 2.0 (`engine/LICENSE`, cópia integral do
  upstream). Atribuição completa em `NOTICE` na raiz do repositório.

O frontend web do colibri (`web/`) foi movido para
[`apps/web-chat/`](../apps/web-chat/) neste repositório, com README
próprio, em vez de ficar dentro de `engine/`.

## O que tem aqui

- `c/glm.c`, `c/olmoe.c` — motores de forward (GLM-family MoE, OLMoE).
- `c/st.h`, `c/json.h`, `c/tok.h`, `c/tok_unicode.h`, `c/compat.h`,
  `c/tier.h` — headers de suporte (safetensors, JSON, tokenizer BPE
  byte-level, compat POSIX/Windows, seleção de tier de hardware).
- `c/backend_cuda.cu` / `c/backend_cuda.h` — backend CUDA opt-in
  (`CUDA=1`), não usado no caminho padrão CPU-only.
- `c/doctor.py`, `c/resource_plan.py`, `c/openai_server.py` — ferramentas
  Python de diagnóstico, planejamento de recursos e servidor
  OpenAI-compatible.
- `c/coli` — script Python wrapper de CLI (não é binário compilado).
- `c/Makefile` — build nativo do motor (`make`, `make test`, `make
  portable`, `make check`). Ver seção "Build" abaixo.
- `c/tests/` — testes C (`test_json.c`, `test_st.c`, `test_tier.c`, etc.)
  e Python (`test_doctor.py`, `test_openai_server.py`,
  `test_resource_plan.py`).
- `c/tools/` — scripts de conversão/benchmark/eval (conversão fp8→int4,
  conversão OLMoE, download de modelos, geração de tabelas Unicode, etc.).
- `c/scripts/` — `run.sh` e `supervisor.sh` de operação.

## O que foi modificado nesta issue (#117)

Nada funcionalmente. Esta issue é vendoring puro:

- Copiado `c/` do fork inalterado.
- `web/` movido (não copiado dentro de `engine/`) para `apps/web-chat/`.
- Nenhum binário compilado commitado (`c/coli` é um script Python, não um
  executável binário — verificado com `file` antes de decidir manter).
- Adicionado este README e o alvo de build no CMake raiz.

Qualquer mudança funcional futura no motor (correção de bug, feature nova,
GGUF, etc.) é escopo de issues subsequentes da épica #116 e deve ser feita
como commits normais dentro deste repositório, documentando divergência do
upstream na seção "Divergência do upstream" abaixo (a criar quando a
primeira mudança acontecer).

## Build

O motor builda via seu próprio `Makefile`, delegado por um
`add_custom_target` no CMake raiz — o Makefile não foi reescrito em CMake
puro (ver ADR-010).

```bash
# via CMake (delega ao Makefile do motor)
cmake --build build --target engine-colibri

# direto, sem CMake
cd engine/c
make glm            # build do binário GLM-family MoE
make test           # testes C + Python do motor
make check           # clean + build portável + testes
```

`make test` inclui tokenizer e regressões dos quantizadores/kernels
int8/int4/int2 sem dependências Python externas. A suíte de oráculos de
forward commitados também integra o CTest do repositório:

```bash
ctest --test-dir build -L engine --output-on-failure
```

Ela valida teacher-forcing, geração greedy e paridade dos caminhos
absorption, DSA-full e speculative decoding. Ausência de ferramenta,
fixture commitada inválida ou mismatch são falhas. Apenas checkpoint de
família ainda não disponível pode produzir um `SKIP checkpoint ...`
explícito. O procedimento e as seeds para regenerar os oráculos ficam em
[`tests/fixtures/engine/README.md`](../tests/fixtures/engine/README.md).

Mudança no forward sem atualizar/rodar esta suíte deve ser rejeitada.

O motor seleciona o chat template a partir de `config.json` e
`tokenizer_config.json`. GLM, DeepSeek e Kimi compartilham a definição
versionada em `c/chat_templates.json`; família ausente ou ambígua interrompe
o startup em vez de assumir GLM. `make test` executa os goldens de prompt, os
vetores adversariais (44 por família), round-trip do tokenizer e streaming
UTF-8 parcial.

Os tokenizers de contrato commitados são pequenos e sintéticos para manter a
suíte offline. Eles não substituem a prova contra checkpoints publicados.
Regere os vetores reais com `c/tools/make_tokenizer_vectors.py`; o arquivo
resultante inclui os hashes SHA-256 do snapshot de referência.

Requisitos do motor (não confundir com os requisitos do runtime C++ em
`runtime/`): compilador C (clang no macOS, gcc no Linux/Windows/MSYS2),
`make`, Python 3 para os testes Python e as ferramentas em `c/tools/`.
CUDA é opt-in (`make CUDA=1 ...`) e não faz parte do caminho padrão
CPU-only alvo deste produto (16GB RAM, sem GPU).

## Como sincronizar com upstream

1. Atualizar o clone local do fork:
   ```bash
   git -C <path-to-fork-clone> fetch origin
   git -C <path-to-fork-clone> log -1 origin/main
   ```
2. Comparar o commit atual documentado em `NOTICE` (raiz deste repo) com o
   commit alvo do fork.
3. Copiar apenas os arquivos alterados de `c/` (e `web/` se aplicável) para
   `engine/c/` (e `apps/web-chat/`), preservando qualquer patch local feito
   neste repositório desde o último sync — resolver conflitos manualmente,
   arquivo por arquivo.
4. Atualizar o commit vendorizado documentado em `NOTICE` e nesta seção.
5. Rodar `make test` em `engine/c/` e a suíte de regressão deste
   repositório (`ctest --test-dir build`) antes de commitar o sync.
6. Registrar o sync como um commit `chore(engine): sync colibri vendor to
   <commit>` separado de qualquer mudança funcional.

Não existe automação de sync (submodule, script de mirror) nesta issue —
é processo manual documentado aqui, por decisão do ADR-010 (Alternativa B
rejeitada).
