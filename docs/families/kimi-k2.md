# Família Kimi K2 no motor CPU

Esta matriz compara o `config.json` publicado de Kimi K2 Instruct com o
caminho DeepSeek-V3 já usado pelo motor C. A fixture local reduz matrizes e
experts, mas preserva o contrato de routing da família; ela não contém nem
substitui o checkpoint K2.

| Área | Kimi K2 publicado | Relação com DeepSeek-V3 | Cobertura local |
| --- | --- | --- | --- |
| Arquitetura | `DeepseekV3ForCausalLM`, `model_type=kimi_k2` | Reusa MLA + MoE sigmoid/noaux | O parser identifica `kimi_k2` e o forward chama o mesmo seletor C testado |
| Camadas/dense | 61 camadas, hidden 7168, primeira camada densa | Dimensões e fronteira densa próprias | `CONFIG_ONLY=1` valida 61/7168 sem carregar pesos; fixture usa 3/64 e uma camada densa |
| Experts | 384 routed, top-8, um shared | Mais experts que DeepSeek-V3 | Limite do motor aceita 384; harness C executa top-8 sobre 384 experts |
| Grupos | `n_group=1`, `topk_group=1` | K2 publicado não aplica group limiting | A fixture mantém 1/1; o mesmo seletor continua cobrindo group limiting de DeepSeek |
| Router | `scoring_func=sigmoid`, `topk_method=noaux_tc`, correção só na seleção, normalização top-k | Mesma semântica-base do router DeepSeek | Config incompatível é rejeitado; vetor independente em `ref_kimi.json` valida índices e pesos sem bias |
| MLA | 64 heads, q-LoRA 1536, kv-LoRA 512, qk 128+64, value 128 | Mesmas projeções, dimensões K2 | Dimensões publicadas atravessam o choke point; tiny reduz proporcionalmente |
| RoPE/contexto | theta 50000 na raiz, YaRN fator 32, contexto 131072 (origem 4096) | K2 não usa `rope_parameters.rope_theta` | Parser agora lê theta raiz; execução acima de 32k e paridade YaRN continuam fora do escopo |
| Vocabulário | 163840, BOS 163584, EOS 163585 | Tokenizer próprio | Limites aceitam 163840; tokenizer/chat template são escopo da #123 |
| Pesos | FP8 e4m3, bloco 128x128, ativações dinâmicas; tronco também pode aparecer BF16 | Nomes de tensores DeepSeek compatíveis | Conversor detecta `kimi_k2`, lê 61 camadas do config e aceita FP8/BF16 shard a shard |

## Fixture e oráculos

- `engine/c/fixtures/kimi_k2_tiny/config.json`: configuração pequena com 16
  experts, top-8, um grupo, MLA reduzida, theta 50000 e metadados FP8. O
  tiny usa RoPE default para que o oráculo curto não finja cobrir YaRN.
- `engine/c/ref_kimi.json`: vetor independente e determinístico do router
  sigmoid/noaux_tc. O bias de correção altera a seleção, mas nunca o peso
  retornado.
- `engine/c/tools/make_kimi_oracle.py`: gera pesos aleatórios pequenos a
  partir da implementação `DeepseekV3ForCausalLM`, sem baixar checkpoint, e
  acrescenta ao `ref_kimi.json` os vetores de teacher forcing e geração.
- `engine/c/tests/test_kimi_family.py`: testa config tiny e dimensões
  publicadas, erro de contrato, routing C, detecção do conversor, escrita
  atômica e, quando o stack opcional está instalado, TF 32/32 e geração
  20/20.

O oráculo de tokens não é pré-gerado neste repositório: isso evitaria
commitar pesos gerados ou apresentar um vetor produzido pelo próprio motor
como referência independente. Na ausência de Torch, Transformers e
Safetensors, esse teste emite skip explícito.

## Conversão resumable

O conversor usa `model_type` e `num_hidden_layers` do `config.json`; K2 não
precisa mais passar `--n-layers 61` manualmente. Cada shard de saída é salvo
primeiro como `.partial` e só então renomeado atomicamente. Um
`out-NNNNN.safetensors` completo é o marcador de retomada, de modo que uma
interrupção não transforma saída parcial em shard concluído.

```bash
python3 tools/convert_fp8_to_int4.py \
  --repo moonshotai/Kimi-K2-Instruct \
  --outdir /Volumes/K2/kimi-k2-int4 \
  --ebits 4 --io-bits 8 --min-free-gb 25
```

O fluxo remoto baixa um shard, converte, remove o original e continua. Não
requer espaço duplicado para o checkpoint inteiro, mas o container final
continua exigindo aproximadamente 550 GB; reserve ao menos 600 GB livres.

## Limites não verificados

- Nenhum checkpoint Kimi foi baixado ou carregado neste trabalho.
- O token-exact 32/32 + 20/20 depende da geração opcional da fixture e ainda
  precisa ser executado no ambiente com as bibliotecas de referência.
- A implementação atual lê theta 50000, mas não reivindica paridade YaRN
  acima de 32k.
- RSS real, coerência de 20 tokens, tok/s, hit-rate e comparação com
  GLM-5.2/DeepSeek numa máquina física de 16 GB continuam sem evidência.
- macOS/Apple Silicon não foi usado nesta validação.
