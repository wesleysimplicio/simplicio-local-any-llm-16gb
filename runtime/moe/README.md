# MoE

Home de routing, paging e telemetria agregada para adapters sparse.

## Estado atual

- `Router` agora executa top-k com softmax e preserva um `RouterDecision`
  consultavel com `entropy`, `load_balance`, `selected_mass` e
  `total_experts`.
- `DeepSeekMoEAdapter` e `KimiMoEAdapter` projetam esse roteamento leve para o
  `GenerationResult` nativo como:
  - `moe_selected_experts`
  - `moe_router_entropy`
  - `moe_load_balance`
  - `moe_selected_mass`
- `DeepSeekMoEAdapter` tambem usa a rota selecionada para materializar uma
  assinatura textual `moe-route eX eY`, deixando visivel no output nativo quais
  experts foram escolhidos para aquele prompt.
- `KimiMoEAdapter` segue a mesma ideia com a assinatura `kimi-route eX eY`,
  mantendo o pager observavel para prompts repetidos e para mudancas de rota.
- `MiniMaxMoEAdapter` amplia a familia MoE advanced com a assinatura
  `minimax-route eX eY`, priorizando prompts com pistas de visao, audio,
  fusao e contexto amplo.
- O contrato de loader agora preserva tambem:
  - `moe_shard_count`
  - `moe_active_experts`
  - `moe_lazy_load`
- Esses campos sao derivados de `ModelAsset` e ficam visiveis no output do CLI
  para manifests/binarios MoE com shards declarados.
- A camada de telemetria tambem deriva:
  - `moe_hit_rate`
  - `moe_eviction_rate`
  - `moe_router_entropy`
- `ExpertPager` agora projeta tambem:
  - `moe_pager_loads`
  - `moe_pager_evictions`
  - `moe_pager_reuses`
  - `moe_resident_experts`
- Lazy load por shard e residency mais rica ainda pertencem aos proximos
  slices.
