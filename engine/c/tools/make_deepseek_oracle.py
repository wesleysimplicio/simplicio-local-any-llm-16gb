"""Constroi um DeepSeek-V3 (arquitetura real `transformers.DeepseekV3ForCausalLM`) MINUSCULO
a pesos random como ORACOLO -- issue #120 (epica #116).

Mesmo padrao de tools/make_glm_oracle.py: MLA (q/kv-LoRA) + router sigmoid/noaux_tc +
shared expert + expert routed, mas com **n_group>1 / topk_group>1 reais** -- e' exatamente
a peca que glm.c nao sabia processar antes desta issue (assumia n_group=1, i.e. GLM-5.2).

Dimensoes reduzidas proporcionalmente ao DeepSeek-V3 real (n_group=8, topk_group=4,
n_routed_experts=256 -> aqui n_group=4, topk_group=2, n_routed_experts=8: mesma razao
n_group/n_experts=1/2 do real 8/256... na verdade o real e' 8/256=1/32; aqui usamos uma
razao maior de proposito para que CADA grupo (2 experts/grupo) e' pequeno o bastante para
exercitar o group-limiting de forma determinstica com poucos experts, mantendo
topk_group<n_group (2<4) para provar que o mascaramento de grupo de fato filtra grupos,
nao apenas topa com n_group==topk_group). Documentado tambem em docs/families/deepseek.md.

DeepSeek-V3 NAO tem DSA indexer (isso e' exclusivo de GLM-5.2/glm_moe_dsa) nem MTP neste
tiny fixture -- o motor C detecta a ausencia automaticamente (has_dsa=has_mtp=0) e cai no
caminho MLA denso + MoE routed/shared, o mesmo caminho ja validado por ref_glm.json.
"""
import json
import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

torch.manual_seed(4202)

cfg = DeepseekV3Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=64,          # MLP densa (primi 2 layer)
    moe_intermediate_size=32,      # expert
    num_hidden_layers=4,           # 2 densi + 2 sparse
    first_k_dense_replace=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    n_group=4,                     # <-- issue #120: n_group>1 real (GLM-5.2 so' tem n_group=1)
    topk_group=2,                  # seleciona 2 dos 4 grupos (2 experts/grupo)
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=24,
    qk_rope_head_dim=8,
    v_head_dim=32,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    rope_interleave=True,
    tie_word_embeddings=False,
    rms_norm_eps=1e-5,
    attention_bias=False,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = DeepseekV3ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    # bias de correcao do router: valores distintos por grupo para que o group-limiting
    # de fato descarte experts fortes que caem num grupo nao selecionado (se todos os bias
    # fossem iguais, o teste nao provaria nada sobre o mascaramento de grupo).
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.15, 0.15, cfg.n_routed_experts))

print("=== tensori dello state_dict (nomi per il loader C) ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
ids = torch.tensor([prompt])
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
print("\nprompt:", prompt)
print("full  :", full)

with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

model.save_pretrained("deepseek_tiny", safe_serialization=True)
cfg_dict = cfg.to_dict()
json.dump(cfg_dict, open("deepseek_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_deepseek.json", "w"))
print("\nsalvato: deepseek_tiny/ (pesi+config) e ref_deepseek.json")
