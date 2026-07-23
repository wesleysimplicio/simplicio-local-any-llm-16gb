import json
import tempfile
import unittest
from pathlib import Path

from chat_templates import (ChatTemplateError, ChatTemplateRegistry,
                            DEFAULT_TEMPLATE_PATH)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ChatGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = ChatTemplateRegistry()
        cls.vectors = json.loads(
            (FIXTURES / "chat_vectors.json").read_text(encoding="utf-8"))

    def test_all_family_goldens(self):
        families = set()
        for case in self.vectors["cases"]:
            families.add(case["family"])
            actual = self.registry.render(
                case["messages"],
                case["family"],
                case["enable_thinking"],
                case["reasoning_effort"],
                case["add_generation_prompt"],
            )
            self.assertEqual(actual, case["expected"], case["name"])
        self.assertEqual(families, {"glm", "deepseek", "kimi"})

    def test_contract_is_deterministic(self):
        first = ChatTemplateRegistry(DEFAULT_TEMPLATE_PATH)
        second = ChatTemplateRegistry(DEFAULT_TEMPLATE_PATH)
        messages = [{"role": "user", "content": "Olá 🙂"}]
        for family in ("glm", "deepseek", "kimi"):
            self.assertEqual(first.render(messages, family),
                             second.render(messages, family))

    def test_thinking_fails_closed_for_unsupported_family(self):
        with self.assertRaisesRegex(ChatTemplateError, "not supported"):
            self.registry.render([{"role": "user", "content": "Hi"}],
                                 "kimi", enable_thinking=True)


class FamilyDetectionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)
        self.registry = ChatTemplateRegistry()

    def tearDown(self):
        self.directory.cleanup()

    def write_config(self, **values):
        (self.model / "config.json").write_text(
            json.dumps(values), encoding="utf-8")

    def test_detects_each_supported_family(self):
        cases = (
            ({"model_type": "glm_moe_dsa"}, "glm"),
            ({"architectures": ["DeepseekV3ForCausalLM"]}, "deepseek"),
            ({"model_type": "kimi_k2"}, "kimi"),
        )
        for config, expected in cases:
            with self.subTest(expected):
                self.write_config(**config)
                self.assertEqual(self.registry.detect(self.model), expected)

    def test_unknown_family_fails_closed(self):
        self.write_config(model_type="llama")
        with self.assertRaisesRegex(ChatTemplateError, "unknown"):
            self.registry.detect(self.model)

    def test_ambiguous_family_fails_closed(self):
        self.write_config(model_type="glm_deepseek")
        with self.assertRaisesRegex(ChatTemplateError, "ambiguous"):
            self.registry.detect(self.model)

    def test_missing_metadata_fails_closed(self):
        with self.assertRaisesRegex(ChatTemplateError, "unknown"):
            self.registry.detect(self.model)

    def test_non_object_config_fails_closed(self):
        (self.model / "config.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ChatTemplateError, "must be an object"):
            self.registry.detect(self.model)


if __name__ == "__main__":
    unittest.main()
