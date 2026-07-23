"""CPU-engine coverage for the Kimi K2 family contract (issue #121)."""

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENGINE_DIR = HERE.parent
GLM_BIN = ENGINE_DIR / "glm"
TINY_CONFIG = ENGINE_DIR / "fixtures" / "kimi_k2_tiny" / "config.json"
REF_PATH = ENGINE_DIR / "ref_kimi.json"
CONVERTER_PATH = ENGINE_DIR / "tools" / "convert_fp8_to_int4.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_fp8_to_int4", CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KimiConfigAndRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["make", "-B", "glm"],
            cwd=ENGINE_DIR,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise unittest.SkipTest(f"engine build unavailable: {result.stderr[-600:]}")

    def _config_only(self, config):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
            return subprocess.run(
                [str(GLM_BIN), "1", "16", "16"],
                cwd=ENGINE_DIR,
                env=dict(os.environ, SNAP=str(snapshot), CONFIG_ONLY="1"),
                capture_output=True,
                text=True,
                timeout=10,
            )

    def test_tiny_fixture_preserves_k2_routing_shape(self):
        config = json.loads(TINY_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["model_type"], "kimi_k2")
        self.assertEqual(config["num_experts_per_tok"], 8)
        self.assertEqual(config["n_group"], 1)
        self.assertEqual(config["topk_group"], 1)
        self.assertEqual(config["topk_method"], "noaux_tc")
        self.assertEqual(config["scoring_func"], "sigmoid")
        self.assertEqual(config["rope_theta"], 50000.0)

        result = self._config_only(config)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("family=kimi_k2", result.stdout)
        self.assertIn("routed_experts=16 topk=8", result.stdout)
        self.assertIn("theta=50000", result.stdout)

    def test_published_k2_dimensions_pass_engine_bounds(self):
        config = json.loads(TINY_CONFIG.read_text(encoding="utf-8"))
        config.update(
            hidden_size=7168,
            intermediate_size=18432,
            kv_lora_rank=512,
            max_position_embeddings=131072,
            moe_intermediate_size=2048,
            n_routed_experts=384,
            num_attention_heads=64,
            num_hidden_layers=61,
            num_key_value_heads=64,
            q_lora_rank=1536,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            vocab_size=163840,
            rope_scaling={
                "beta_fast": 1.0,
                "beta_slow": 1.0,
                "factor": 32.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
        )
        result = self._config_only(config)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("layers=61 hidden=7168 routed_experts=384 topk=8", result.stdout)

    def test_rejects_non_k2_router_contract(self):
        config = json.loads(TINY_CONFIG.read_text(encoding="utf-8"))
        config["topk_method"] = "greedy"
        result = self._config_only(config)
        self.assertGreater(result.returncode, 0)
        self.assertIn("kimi_k2 requer topk_method=noaux_tc", result.stderr)

    def test_rejects_topk_larger_than_expert_pool(self):
        config = json.loads(TINY_CONFIG.read_text(encoding="utf-8"))
        config["num_experts_per_tok"] = 17
        result = self._config_only(config)
        self.assertGreater(result.returncode, 0)
        self.assertIn("excede n_routed_experts=16", result.stderr)

    def test_c_router_matches_independent_reference_vector(self):
        reference = json.loads(REF_PATH.read_text(encoding="utf-8"))
        logits = reference["router_logits"]
        bias = reference["e_score_correction_bias"]
        expected = sorted(
            range(len(logits)),
            key=lambda expert: 1.0 / (1.0 + math.exp(-logits[expert])) + bias[expert],
            reverse=True,
        )[:8]
        self.assertEqual(expected, reference["expected_indices"])

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "route_fixture.c"
            binary = directory / "route_fixture"
            source.write_text(
                """
#include <stdio.h>
#include "moe_route.h"
int main(void){
    float logits[16]={%s};
    float bias[16]={%s};
    float prob[16], choice[16], weights[8];
    int indices[8];
    coli_moe_select(logits,bias,16,8,1,1,prob,choice,NULL,NULL,indices,weights);
    for(int i=0;i<8;i++) printf("%%d %%.9g\\n",indices[i],weights[i]);
    return 0;
}
"""
                % (
                    ",".join(f"{value}f" for value in logits),
                    ",".join(f"{value}f" for value in bias),
                ),
                encoding="utf-8",
            )
            build = subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ENGINE_DIR), str(source), "-lm", "-o", str(binary)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)

        rows = [line.split() for line in run.stdout.splitlines()]
        self.assertEqual([int(row[0]) for row in rows], reference["expected_indices"])
        for row, expected_weight in zip(rows, reference["expected_unscaled_weights"]):
            self.assertAlmostEqual(float(row[1]), expected_weight, places=6)

    def test_c_router_handles_published_384_expert_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "route_384.c"
            binary = directory / "route_384"
            source.write_text(
                """
#include <stdio.h>
#include "moe_route.h"
int main(void){
    float logits[384], bias[384], prob[384], choice[384], weights[8];
    int indices[8];
    for(int e=0;e<384;e++){ logits[e]=(float)(e%17-8)/4.f; bias[e]=(float)e/10000.f; }
    for(int rank=0;rank<8;rank++){
        int expert=300+rank; logits[expert]=1.f+(float)rank/10.f; bias[expert]=2.f;
    }
    coli_moe_select(logits,bias,384,8,1,1,prob,choice,NULL,NULL,indices,weights);
    for(int rank=0;rank<8;rank++) printf("%d\\n",indices[rank]);
    return 0;
}
""",
                encoding="utf-8",
            )
            build = subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ENGINE_DIR), str(source), "-lm", "-o", str(binary)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual([int(line) for line in run.stdout.splitlines()], list(range(307, 299, -1)))


class KimiConverterTest(unittest.TestCase):
    def test_converter_detects_k2_and_uses_config_layer_count(self):
        converter = _load_converter()
        layers, family = converter.resolve_model_config(TINY_CONFIG.parent)
        self.assertEqual((layers, family), (3, "kimi_k2"))
        self.assertEqual(
            converter.classify("model.layers.2.mlp.experts.15.gate_proj.weight", layers),
            "x",
        )
        self.assertEqual(
            converter.classify("model.layers.3.mlp.experts.0.gate_proj.weight", layers),
            "skip",
        )

    def test_atomic_output_is_the_resume_marker(self):
        converter = _load_converter()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out-00000.safetensors"

            def fake_save(tensors, path):
                Path(path).write_bytes(tensors["payload"])

            converter.save_file_atomic(fake_save, {"payload": b"complete"}, str(output))
            first_mtime = output.stat().st_mtime_ns
            self.assertEqual(output.read_bytes(), b"complete")
            self.assertFalse(Path(str(output) + ".partial").exists())
            time.sleep(0.001)
            if output.exists():
                resumed = True
            else:
                converter.save_file_atomic(fake_save, {"payload": b"rewritten"}, str(output))
                resumed = False
            self.assertTrue(resumed)
            self.assertEqual(output.stat().st_mtime_ns, first_mtime)


@unittest.skipUnless(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("transformers")
    and importlib.util.find_spec("safetensors"),
    "torch/transformers/safetensors unavailable: full tiny K2 TF/generation oracle not generated",
)
class KimiTokenOracleTest(unittest.TestCase):
    def test_teacher_forcing_and_generation_against_generated_oracle(self):
        oracle_script = ENGINE_DIR / "tools" / "make_kimi_oracle.py"
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "kimi_tiny"
            reference = Path(directory) / "ref_kimi.json"
            generated = subprocess.run(
                [sys.executable, str(oracle_script), "--output-dir", str(snapshot),
                 "--reference", str(reference)],
                cwd=ENGINE_DIR,
                capture_output=True,
                text=True,
                timeout=240,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            for teacher_forcing, marker in (
                (True, "32/32"),
                (False, "Token coincidenti: 20/20"),
            ):
                env = dict(os.environ, SNAP=str(snapshot), REF=str(reference))
                if teacher_forcing:
                    env["TF"] = "1"
                result = subprocess.run(
                    [str(GLM_BIN), "64", "16", "16"],
                    cwd=ENGINE_DIR,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(marker, result.stdout)


if __name__ == "__main__":
    unittest.main()
