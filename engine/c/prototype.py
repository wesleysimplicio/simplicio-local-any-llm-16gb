#!/usr/bin/env python3
"""CLI for the Prototype-First local worker boundary."""

import argparse
import json
import sys

from prototype_worker import (ARTIFACT_FIELDS, WorkerPolicy, evaluate,
                              generate_without_runtime, manifest)


def parse_candidate(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid candidate JSON: {error.msg}") from error


def parser():
    result = argparse.ArgumentParser(prog="us4-cli prototype")
    commands = result.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    for name in ("generate", "critic", "judge", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--artifact-type", choices=sorted(ARTIFACT_FIELDS),
                             required=True)
        command.add_argument("--candidate-json")
        command.add_argument("--quality-floor", type=float, default=0.75)
        command.add_argument("--hard-rss-bytes", type=int,
                             default=13_000_000_000)
        command.add_argument("--candidate-model")
        command.add_argument("--judge-model")
        command.add_argument("--allow-local-judge", action="store_true")
        command.add_argument("--json", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        report = manifest()
    else:
        policy = WorkerPolicy(args.hard_rss_bytes, args.quality_floor,
                              allow_local_judge=args.allow_local_judge)
        if args.command == "generate" and args.candidate_json is None:
            report = generate_without_runtime(args.artifact_type, policy)
        elif args.candidate_json is None:
            raise ValueError("--candidate-json is required")
        else:
            candidate = parse_candidate(args.candidate_json)
            role = "judge" if args.command == "judge" else "critic"
            report = evaluate(
                args.artifact_type, candidate, policy,
                candidate_model=args.candidate_model,
                judge_model=args.judge_model, role=role,
            )
    print(json.dumps(report, sort_keys=True) if getattr(args, "json", False)
          else json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") in (None, "accept") else 78


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"us4-cli prototype: {error}", file=sys.stderr)
        raise SystemExit(64)
