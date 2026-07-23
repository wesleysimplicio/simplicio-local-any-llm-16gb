"""Family-specific chat templates backed by one deterministic JSON contract."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_PATH = HERE / "chat_templates.json"
SUPPORTED_ROLES = ("system", "developer", "user", "assistant")


class ChatTemplateError(ValueError):
    pass


def _load_json(path):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChatTemplateError(f"cannot load chat template contract: {error}") from error
    if not isinstance(document, dict):
        raise ChatTemplateError(f"JSON document must be an object: {path}")
    return document


class ChatTemplateRegistry:
    def __init__(self, path=DEFAULT_TEMPLATE_PATH):
        self.path = Path(path)
        document = _load_json(self.path)
        if document.get("schema_version") != 1:
            raise ChatTemplateError("unsupported chat template schema")
        families = document.get("families")
        if not isinstance(families, dict) or not families:
            raise ChatTemplateError("chat template contract has no families")
        self.families = families
        for family, template in families.items():
            self._validate(family, template)

    @staticmethod
    def _validate(family, template):
        required_strings = (
            "bos", "default_system", "system_mode", "system_separator",
            "reasoning_effort_format", "generation_prompt",
            "thinking_generation_prompt",
        )
        if not isinstance(template, dict):
            raise ChatTemplateError(f"invalid template for {family}")
        if any(not isinstance(template.get(name), str) for name in required_strings):
            raise ChatTemplateError(f"template {family} has invalid string fields")
        if template["system_mode"] not in ("role", "prefix"):
            raise ChatTemplateError(f"template {family} has invalid system_mode")
        if not isinstance(template.get("supports_thinking"), bool):
            raise ChatTemplateError(f"template {family} has invalid supports_thinking")
        aliases = template.get("aliases")
        if not isinstance(aliases, list) or not aliases or not all(
                isinstance(alias, str) and alias for alias in aliases):
            raise ChatTemplateError(f"template {family} has invalid aliases")
        roles = template.get("roles")
        if not isinstance(roles, dict) or set(roles) != set(SUPPORTED_ROLES):
            raise ChatTemplateError(f"template {family} has invalid roles")
        for role, rule in roles.items():
            if not isinstance(rule, dict):
                raise ChatTemplateError(f"template {family} has invalid role {role}")
            if any(not isinstance(rule.get(name), str)
                   for name in ("prefix", "suffix", "strip_after")):
                raise ChatTemplateError(f"template {family} has invalid role {role}")
            if not isinstance(rule.get("trim"), bool):
                raise ChatTemplateError(f"template {family} has invalid role {role}")

    def get(self, family):
        template = self.families.get(family)
        if template is None:
            raise ChatTemplateError(f"unsupported model family: {family!r}")
        return template

    def detect(self, model_dir):
        model_dir = Path(model_dir)
        fingerprints = []
        for filename in ("config.json", "tokenizer_config.json"):
            path = model_dir / filename
            if not path.is_file():
                continue
            document = _load_json(path)
            for key in ("model_type", "tokenizer_class", "name_or_path"):
                value = document.get(key)
                if isinstance(value, str):
                    fingerprints.append(value.lower())
            architectures = document.get("architectures")
            if isinstance(architectures, list):
                fingerprints.extend(value.lower() for value in architectures
                                    if isinstance(value, str))
        matches = []
        for family, template in self.families.items():
            if any(alias.lower() in value for alias in template["aliases"]
                   for value in fingerprints):
                matches.append(family)
        if len(matches) != 1:
            detail = "unknown" if not matches else f"ambiguous ({', '.join(matches)})"
            raise ChatTemplateError(f"model family is {detail}; refusing implicit template")
        return matches[0]

    def render(self, messages, family, enable_thinking=False,
               reasoning_effort=None, add_generation_prompt=True):
        template = self.get(family)
        if not isinstance(messages, list) or not messages:
            raise ChatTemplateError("messages must be a non-empty array")
        if enable_thinking and not template["supports_thinking"]:
            raise ChatTemplateError(f"thinking is not supported for family {family}")

        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ChatTemplateError(f"message {index} must be an object")
            role = message.get("role")
            if role not in SUPPORTED_ROLES:
                raise ChatTemplateError(f"unsupported message role: {role!r}")
            content = message.get("content")
            if not isinstance(content, str):
                raise ChatTemplateError(f"message {index} content must be text")
            normalized.append((role, content))

        prompt = [template["bos"]]
        system_messages = [content for role, content in normalized
                           if role in ("system", "developer")]
        if template["system_mode"] == "prefix":
            prompt.append(template["system_separator"].join(system_messages))
        elif (normalized[0][0] not in ("system", "developer") and
              template["default_system"]):
            rule = template["roles"]["system"]
            prompt.extend((rule["prefix"], template["default_system"], rule["suffix"]))

        if enable_thinking and template["reasoning_effort_format"]:
            effort = "High" if reasoning_effort == "high" else "Max"
            rule = template["roles"]["system"]
            text = template["reasoning_effort_format"].replace("{effort}", effort)
            prompt.extend((rule["prefix"], text, rule["suffix"]))

        for role, content in normalized:
            if template["system_mode"] == "prefix" and role in ("system", "developer"):
                continue
            rule = template["roles"][role]
            marker = rule["strip_after"]
            if marker and marker in content:
                content = content.rsplit(marker, 1)[1]
            if rule["trim"]:
                content = content.strip()
            prompt.extend((rule["prefix"], content, rule["suffix"]))

        if add_generation_prompt:
            prompt.append(template["thinking_generation_prompt"] if enable_thinking
                          else template["generation_prompt"])
        return "".join(prompt)


REGISTRY = ChatTemplateRegistry()


def detect_model_family(model_dir):
    return REGISTRY.detect(model_dir)


def render_family_chat(messages, family, enable_thinking=False,
                       reasoning_effort=None, add_generation_prompt=True):
    return REGISTRY.render(messages, family, enable_thinking, reasoning_effort,
                           add_generation_prompt)
