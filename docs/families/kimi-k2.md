# Kimi K2 family matrix

This document records the Kimi K2 work that is known from the repository and
the source issue. It deliberately separates offline contract coverage from
real-model execution; the fixture in `tests/fixtures/models/kimi-k2-instruct`
is not a Kimi K2 checkpoint.

| Area | Kimi K2 target | DeepSeek-V3 relationship | Current repository state |
| --- | --- | --- | --- |
| Architecture | DeepSeek-V3-derived MLA + MoE | Same broad family, with different dimensions and routing limits | The runtime has a Kimi adapter and a separate DeepSeek adapter; both currently use the runtime's bounded contract path. |
| Routed experts | 384 routed experts, 8 active plus 1 shared | DeepSeek-V3 uses a different expert count/top-k | Kimi routing is exercised by the existing offline adapter contract, but the real 384-expert checkpoint has not been loaded here. |
| Dense trunk | 61 layers, hidden size 7168 | Dimensions are family-specific | No Kimi K2 `config.json` or real weights are vendored in this repository. |
| Vocabulary | Approximately 163k tokens | Family tokenizer differs | The fixture tokenizer is intentionally tiny and only validates tokenizer metadata/contract plumbing. |
| Quantization | Published checkpoint may use QAT; exact release format must be detected | Conversion rules are checkpoint-specific | No real K2 shard conversion or QAT loss measurement has been run. |
| Context/RoPE | Long-context scaling and YaRN/static choices must be read from the actual config | Do not copy DeepSeek values blindly | No real K2 config is present to validate these values. |

## Implemented offline

- The Kimi model manifest resolves `tokenizer.json` and records a deterministic
  fixture seed and vocabulary.
- The runtime exposes tokenizer metadata and chat-template fields for family
  assets, and the Kimi fixture is covered by tokenizer/model-asset contracts.
- The Kimi adapter routes a deterministic tiny request through the same
  offline contract surface as the other MoE families.

## Still required for issue #121

- Obtain and inspect the real K2 `config.json`, tokenizer and checkpoint shard
  index; record every dimension difference here before conversion.
- Add a shape-preserving tiny K2 oracle and run teacher forcing 32/32 plus
  greedy generation 20/20 against an independently generated reference.
- Validate resumable shard-to-container conversion with synthetic K2-shaped
  shards and then with a real checkpoint.
- Run the real model for at least 20 coherent tokens on a 16 GB host with the
  required approximately 600 GB of free SSD, and compare resident trunk,
  tok/s and cache hit-rate against the other target families.

## Explicit blockers

This Windows host has neither the real K2 checkpoint nor a 16 GB/600 GB test
machine. No real Kimi execution, throughput, hit-rate, conversion result, or
token-exact K2 oracle is claimed by this document.
