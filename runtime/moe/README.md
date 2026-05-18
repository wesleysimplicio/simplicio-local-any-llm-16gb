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
- `ExpertPager` agora projeta tambem:
  - `moe_pager_loads`
  - `moe_pager_evictions`
  - `moe_pager_reuses`
  - `moe_resident_experts`
- Lazy load por shard e residency mais rica ainda pertencem aos proximos
  slices.
