"""Issue #120 (epica #116): roteamento MoE group-limited (n_group>1) da familia
DeepSeek-V3/R1, validado contra um oraculo REAL `transformers.DeepseekV3ForCausalLM`
(arquitetura real, pesos aleatorios, escala minuscula) -- NAO um checkpoint DeepSeek-V3/R1
real (~650GB) e NAO uma maquina real de 16GB (issue #118, bloqueada por hardware). Ver
docs/families/deepseek.md para a semantica do router e a matriz de diferencas vs GLM-5.2.

Se torch/transformers nao estiverem instalados, ou o motor nao builda neste ambiente, os
testes sao pulados com uma razao explicita (nao skipados silenciosamente).
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE_DIR = HERE.parent
GLM_BIN = ENGINE_DIR / "glm"
ORACLE_SCRIPT = ENGINE_DIR / "tools" / "make_deepseek_oracle.py"
CONVERT_SCRIPT = ENGINE_DIR / "tools" / "convert_fp8_to_int4.py"
DEEPSEEK_TINY = ENGINE_DIR / "deepseek_tiny"
REF_PATH = ENGINE_DIR / "ref_deepseek.json"


def _have_module(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAVE_ML_STACK = _have_module("torch") and _have_module("transformers") and _have_module("safetensors")


def _build_engine():
    if GLM_BIN.is_file():
        return True
    result = subprocess.run(["make", "glm"], cwd=ENGINE_DIR, capture_output=True, text=True)
    return result.returncode == 0 and GLM_BIN.is_file()


def _ensure_oracle_fixture():
    if DEEPSEEK_TINY.is_dir() and (DEEPSEEK_TINY / "config.json").is_file() and \
       (DEEPSEEK_TINY / "model.safetensors").is_file() and REF_PATH.is_file():
        return True
    result = subprocess.run([sys.executable, str(ORACLE_SCRIPT)], cwd=ENGINE_DIR,
                            capture_output=True, text=True, timeout=180)
    return result.returncode == 0 and (DEEPSEEK_TINY / "model.safetensors").is_file()


@unittest.skipUnless(HAVE_ML_STACK, "torch/transformers/safetensors indisponiveis: fixture "
                     "sintetica nao pode ser gerada. Rode 'pip install torch transformers "
                     "safetensors' para exercitar estes testes localmente (nao requer o "
                     "checkpoint DeepSeek-V3/R1 real nem hardware de 16GB).")
class DeepseekGroupLimitedRoutingTest(unittest.TestCase):
    """Baseado na fixture do oraculo (make_deepseek_oracle.py): arquitetura DeepseekV3
    real (transformers), pesos aleatorios, minuscula, mas com n_group=4/topk_group=2 REAIS
    -- o group-limiting desta issue esta' de fato ativo e sendo testado."""

    @classmethod
    def setUpClass(cls):
        if not _build_engine():
            raise unittest.SkipTest("nao foi possivel compilar engine/c/glm (make glm falhou)")
        if not _ensure_oracle_fixture():
            raise unittest.SkipTest("nao foi possivel gerar engine/c/deepseek_tiny "
                                     "(make_deepseek_oracle.py falhou)")

    def _run_glm(self, snap, ref=REF_PATH, extra_env=None, cap=64, ebits=16, dbits=16, tf=False, timeout=60):
        env = dict(os.environ, SNAP=str(snap), REF=str(ref))
        if tf:
            env["TF"] = "1"
        if extra_env:
            env.update(extra_env)
        return subprocess.run([str(GLM_BIN), str(cap), str(ebits), str(dbits)],
                              cwd=ENGINE_DIR, env=env, capture_output=True, text=True, timeout=timeout)

    def test_config_declares_real_group_limiting(self):
        """Sanity check da fixture: precisa exercitar n_group>1 e topk_group<n_group, senao
        o teste nao provaria nada sobre o group-limiting (issue #120)."""
        cfg = json.loads((DEEPSEEK_TINY / "config.json").read_text())
        self.assertGreater(cfg["n_group"], 1)
        self.assertLess(cfg["topk_group"], cfg["n_group"])
        self.assertEqual(cfg["n_routed_experts"] % cfg["n_group"], 0)

    def test_teacher_forcing_matches_oracle_32_of_32(self):
        """Prefill via teacher-forcing: forward em toda a sequencia (prompt+geracao do
        oraculo) deve reproduzir EXATAMENTE o argmax por posicao do transformers real,
        incluindo o group-limited routing (n_group=4, topk_group=2)."""
        run = self._run_glm(DEEPSEEK_TINY, tf=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("32/32", run.stdout)
        self.assertNotIn("[ORACLE] mismatch", run.stderr)

    def test_greedy_generation_matches_oracle_20_of_20(self):
        """Geracao greedy token-a-token (KV cache real, nao teacher-forcing) deve
        reproduzir exatamente os 20 tokens gerados pelo oraculo transformers."""
        run = self._run_glm(DEEPSEEK_TINY, tf=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Token coincidenti: 20/20", run.stdout)

    def test_group_limiting_is_load_bearing_not_a_coincidence(self):
        """Se a mascara de grupo fosse removida (n_group=1 forcado no config, ignorando o
        group-limiting real do fixture), o teacher-forcing NAO deve mais bater 32/32 --
        prova que a logica desta issue e' realmente exercitada pelo fixture, nao um
        no-op que por acaso da o resultado certo."""
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "deepseek_tiny_ng1"
            shutil.copytree(DEEPSEEK_TINY, broken)
            cfg = json.loads((broken / "config.json").read_text())
            cfg["n_group"] = 1
            cfg["topk_group"] = 1
            (broken / "config.json").write_text(json.dumps(cfg))
            run = self._run_glm(broken, tf=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertNotIn("32/32", run.stdout)

    def test_glm_regression_still_32_of_32_with_n_group_1(self):
        """Zero regressao (AC da issue #120): a fixture GLM-5.2 (ref_glm.json / glm_tiny/,
        n_group=1 implicito) continua bit-identica ao comportamento legado. Requer que
        tools/make_glm_oracle.py ja tenha sido rodado (test_16gb_profile.py garante isso
        no mesmo processo de test discovery)."""
        glm_tiny = ENGINE_DIR / "glm_tiny"
        ref_glm = ENGINE_DIR / "ref_glm.json"
        if not (glm_tiny / "model.safetensors").is_file():
            result = subprocess.run([sys.executable, str(ENGINE_DIR / "tools" / "make_glm_oracle.py")],
                                    cwd=ENGINE_DIR, capture_output=True, text=True, timeout=180)
            if result.returncode != 0 or not (glm_tiny / "model.safetensors").is_file():
                raise unittest.SkipTest("nao foi possivel gerar engine/c/glm_tiny")
        run = self._run_glm(glm_tiny, ref=ref_glm, tf=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("32/32", run.stdout)
        self.assertNotIn("[ORACLE] mismatch", run.stderr)

    def test_converter_produces_valid_deepseek_container(self):
        """tools/convert_fp8_to_int4.py (AC #5): classify()/convert_shard() sao genericos
        por nome de tensor -- a familia DeepSeek usa os MESMOS nomes de peso MLA/MoE de
        GLM-5.2, entao o mesmo conversor produz um container valido (--n-layers precisa
        refletir a contagem real de camadas: 4 neste fixture tiny, 61 no DeepSeek-V3/R1
        real, vs 78 do default GLM-5.2). O motor deve carregar e RODAR esse container sem
        abortar (nao exigimos bit-exatidao em int4 sobre pesos aleatorios -- ruido de
        quantizacao esperado, ja observado tambem em glm_tiny)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "deepseek_tiny_i4"
            result = subprocess.run([sys.executable, str(CONVERT_SCRIPT), "--indir", str(DEEPSEEK_TINY),
                                     "--outdir", str(outdir), "--ebits", "4", "--io-bits", "8",
                                     "--n-layers", "4"],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(any(outdir.glob("out-*.safetensors")))
            run = self._run_glm(outdir, tf=True, ebits=4, dbits=4)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
