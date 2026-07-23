#!/usr/bin/env python3
"""Prototype-First worker contract; inference remains Runtime-governed."""

from dataclasses import dataclass
import hashlib
import json

from backend_contract import PROTOCOL, admission_estimate, capability_probe


PROTOTYPE_PROTOCOL = "simplicio.prototype-worker/v1"
ARTIFACT_FIELDS = {
    "wireframe": frozenset(("name", "screens")),
    "schema": frozenset(("name", "fields")),
    "model": frozenset(("name", "entities")),
    "failing-test": frozenset(("name", "test", "expected_failure")),
    "plan": frozenset(("name", "steps")),
    "prompt-candidate": frozenset(("name", "prompt")),
}


@dataclass(frozen=True)
class WorkerPolicy:
    hard_rss_bytes: int
    quality_floor: float = 0.75
    offline: bool = True
    allow_local_judge: bool = False

    def __post_init__(self):
        if self.hard_rss_bytes <= 0:
            raise ValueError("hard_rss_bytes must be positive")
        if not 0 <= self.quality_floor <= 1:
            raise ValueError("quality_floor must be between 0 and 1")
        if not self.offline:
            raise ValueError("prototype worker is offline-only")


def manifest(repo_root=None):
    backend = capability_probe(repo_root)
    return {
        "protocol": PROTOTYPE_PROTOCOL,
        "backend_protocol": PROTOCOL,
        "offline": True,
        "prompt_logging": False,
        "effect_authority": "none",
        "artifact_types": sorted(ARTIFACT_FIELDS),
        "commands": ["generate", "critic", "judge", "summarize", "doctor"],
        "backend": backend,
    }


def validate_candidate(artifact_type, candidate):
    if artifact_type not in ARTIFACT_FIELDS:
        return False, ["unsupported-artifact-type"]
    if not isinstance(candidate, dict):
        return False, ["candidate-must-be-object"]
    missing = sorted(ARTIFACT_FIELDS[artifact_type] - candidate.keys())
    if missing:
        return False, [f"missing:{name}" for name in missing]
    try:
        encoded = json.dumps(candidate, sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError):
        return False, ["candidate-not-json-safe"]
    if len(encoded) > 1_000_000:
        return False, ["candidate-too-large"]
    return True, []


def confidence(candidate):
    evidence = candidate.get("evidence", []) if isinstance(candidate, dict) else []
    tests = candidate.get("tests", []) if isinstance(candidate, dict) else []
    score = 0.55
    if isinstance(evidence, list) and evidence:
        score += min(0.2, len(evidence) * 0.05)
    if isinstance(tests, list) and tests:
        score += min(0.2, len(tests) * 0.05)
    return min(score, 0.95)


def evaluate(artifact_type, candidate, policy, *, candidate_model=None,
             judge_model=None, role="critic"):
    valid, errors = validate_candidate(artifact_type, candidate)
    score = confidence(candidate) if valid else 0.0
    reasons = list(errors)
    if role == "judge":
        if not policy.allow_local_judge:
            reasons.append("judge-independence-requires-external-runtime")
        elif not judge_model or judge_model == candidate_model:
            reasons.append("judge-model-must-be-independent")
    if score < policy.quality_floor:
        reasons.append("below-quality-floor")
    status = "accept" if not reasons else "escalate_remote"
    digest = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
        if isinstance(candidate, dict) else b"invalid"
    ).hexdigest()
    return {
        "protocol": PROTOTYPE_PROTOCOL,
        "status": status,
        "artifact_type": artifact_type,
        "schema_valid": valid,
        "confidence": score,
        "quality_floor": policy.quality_floor,
        "reasons": reasons,
        "candidate_sha256": digest,
        "effect_authority": "none",
        "prompt_logged": False,
    }


def fit_task(*, model_bytes, dense_bytes, available_memory, available_disk,
             policy):
    return admission_estimate(
        model_bytes=model_bytes,
        dense_bytes=dense_bytes,
        available_memory=available_memory,
        available_disk=available_disk,
        hard_rss_limit=policy.hard_rss_bytes,
        workload="background",
    )


def generate_without_runtime(artifact_type, policy):
    if artifact_type not in ARTIFACT_FIELDS:
        raise ValueError("unsupported artifact type")
    return {
        "protocol": PROTOTYPE_PROTOCOL,
        "status": "escalate_remote",
        "artifact_type": artifact_type,
        "reason": "runtime-inference-lease-required",
        "quality_floor": policy.quality_floor,
        "effect_authority": "none",
        "prompt_logged": False,
    }
