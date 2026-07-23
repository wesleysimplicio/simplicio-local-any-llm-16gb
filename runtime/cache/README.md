# Cache

Home for prefix cache, SSD cold cache, reusable session cache components, and
MoE sparsity-pattern reuse surfaces.

- `SparsityAwareCache` keys entries by `family + expert-pattern hash`.
- MoE adapters use it to surface hit/miss effectiveness without changing
  executable expert selection semantics.
- `MultimodalCache` keys `family + modality + token hash` and is currently
  consumed only by `MiniMaxMoEAdapter`, keeping dense-only paths isolated.

## Cache de experts no perfil 16 GB

`ExpertPager` (em `runtime/moe`) mantém uma política adaptativa mensurável:

- o conjunto pinado é recalculado pelos experts mais tocados, em vez de
  preservar para sempre os primeiros que cruzaram o limiar;
- o runtime faz o rebalance em lotes (a cada 1024 lookups) depois de preencher
  o pin budget, evitando ordenar milhares de experts no hot path;
- evicção continua preferindo residents não pinados;
- `WarmupLearnedPins()` pré-carrega somente o conjunto aprendido e nunca
  ultrapassa a capacidade configurada;
- `UsageHistogram()` e `ExportUsageJson()` expõem frequência, hit/miss,
  concentração de 50% e estado de pin sem registrar prompts.

Os números são contadores do runtime. Tok/s, RSS e estabilidade entre idiomas
continuam exigindo captura no checkpoint e hardware-alvo.
