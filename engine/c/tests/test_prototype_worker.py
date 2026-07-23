import unittest

from prototype_worker import (PROTOTYPE_PROTOCOL, WorkerPolicy, evaluate,
                              fit_task, generate_without_runtime, manifest,
                              validate_candidate)


class PrototypeWorkerTest(unittest.TestCase):
    def setUp(self):
        self.policy = WorkerPolicy(13_000_000_000, quality_floor=0.6)
        self.plan = {
            "name": "worker",
            "steps": ["probe", "admit", "execute"],
            "evidence": ["schema"],
            "tests": ["contract"],
        }

    def test_manifest_is_offline_and_has_no_effect_authority(self):
        report = manifest()
        self.assertEqual(report["protocol"], PROTOTYPE_PROTOCOL)
        self.assertTrue(report["offline"])
        self.assertFalse(report["prompt_logging"])
        self.assertEqual(report["effect_authority"], "none")

    def test_candidate_schema_is_fail_closed(self):
        self.assertEqual(validate_candidate("plan", self.plan), (True, []))
        valid, errors = validate_candidate("plan", {"name": "missing"})
        self.assertFalse(valid)
        self.assertEqual(errors, ["missing:steps"])

    def test_quality_floor_accepts_evidenced_candidate(self):
        report = evaluate("plan", self.plan, self.policy)
        self.assertEqual(report["status"], "accept")
        self.assertTrue(report["schema_valid"])
        self.assertFalse(report["prompt_logged"])

    def test_low_quality_and_invalid_json_escalate(self):
        report = evaluate(
            "plan", {"name": "worker", "steps": []},
            WorkerPolicy(13_000_000_000, quality_floor=0.9),
        )
        self.assertEqual(report["status"], "escalate_remote")
        self.assertIn("below-quality-floor", report["reasons"])

    def test_judge_requires_independence(self):
        report = evaluate(
            "plan", self.plan, self.policy, role="judge",
            candidate_model="local", judge_model="local",
        )
        self.assertEqual(report["status"], "escalate_remote")
        self.assertIn("judge-independence-requires-external-runtime",
                      report["reasons"])
        independent = evaluate(
            "plan", self.plan,
            WorkerPolicy(13_000_000_000, 0.6, allow_local_judge=True),
            role="judge", candidate_model="candidate", judge_model="critic",
        )
        self.assertEqual(independent["status"], "accept")

    def test_fit_task_reuses_governed_admission(self):
        report = fit_task(
            model_bytes=10_000_000_000, dense_bytes=6_000_000_000,
            available_memory=16_000_000_000,
            available_disk=400_000_000_000, policy=self.policy,
        )
        self.assertEqual(report["decision"], "admit")

    def test_generate_without_runtime_never_fakes_success(self):
        report = generate_without_runtime("plan", self.policy)
        self.assertEqual(report["status"], "escalate_remote")
        self.assertEqual(report["reason"], "runtime-inference-lease-required")


if __name__ == "__main__":
    unittest.main()
