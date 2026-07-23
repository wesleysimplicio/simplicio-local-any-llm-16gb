"""Generate a tiny architecture-derived Kimi K2 token oracle without a checkpoint.

The fixture keeps K2's published sigmoid/noaux_tc top-8 routing, one dense
layer, MLA shape, shared expert and root-level RoPE theta. Expert count and
matrix dimensions are reduced so the generated random-weight model is small.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="kimi_tiny")
parser.add_argument("--reference", default="ref_kimi.json")
args = parser.parse_args()

torch.manual_seed(121)

config = DeepseekV3Config(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=16,
    num_hidden_layers=3,
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=16,
    num_experts_per_tok=8,
    n_shared_experts=1,
    n_group=1,
    topk_group=1,
    q_lora_rank=32,
    kv_lora_rank=16,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=16,
    norm_topk_prob=True,
    routed_scaling_factor=2.827,
    rope_parameters={"rope_type": "default", "rope_theta": 50000.0},
    rope_interleave=True,
    tie_word_embeddings=False,
    rms_norm_eps=1e-6,
    attention_bias=False,
    max_position_embeddings=4096,
)
config._attn_implementation = "eager"

model = DeepseekV3ForCausalLM(config).eval()
with torch.no_grad():
    for _, parameter in model.named_parameters():
        if parameter.dim() >= 2:
            parameter.normal_(0, 0.05)
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.15, 0.15, config.n_routed_experts)
            )

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
ids = torch.tensor([prompt])
with torch.no_grad():
    generated = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full_ids = generated[0].tolist()
with torch.no_grad():
    teacher_forcing = model(torch.tensor([full_ids]), use_cache=False).logits[0]
tf_pred = teacher_forcing.argmax(-1).tolist()

output_dir = Path(args.output_dir)
model.save_pretrained(output_dir, safe_serialization=True)
fixture_config = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "kimi_k2_tiny"
        / "config.json"
    ).read_text(encoding="utf-8")
)
(output_dir / "config.json").write_text(
    json.dumps(fixture_config, indent=2), encoding="utf-8"
)

reference_path = Path(args.reference)
base_reference = Path(__file__).resolve().parents[1] / "ref_kimi.json"
reference = json.loads(base_reference.read_text(encoding="utf-8"))
reference.update(
    prompt_ids=prompt,
    full_ids=full_ids,
    tf_pred=tf_pred,
    teacher_forcing="32/32",
    greedy_generation="20/20",
    unverified_reason=None,
)
reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
print(f"generated {output_dir} and {reference_path}")
