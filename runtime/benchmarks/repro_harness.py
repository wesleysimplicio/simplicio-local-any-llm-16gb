#!/usr/bin/env python3
"""Reproducible benchmark capture harness for issue #118/#126."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from string import Template
from typing import Any


SCHEMA_ID = "local-any-llm-16gb/repro-bench-run-v1"
KV_LINE_RE = re.compile(r"^([A-Za-z0-9_.-]+)=(.*)$")


def _physical_memory_bytes() -> int | None:
    system = platform.system().lower()
    if system == "windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return int(state.ullTotalPhys)
        return None

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                return pages * page_size
        except (ValueError, OSError):
            return None
    return None


def probe_host() -> dict[str, Any]:
    memory_bytes = _physical_memory_bytes()
    return {
        "platform_system": platform.system().lower(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
        "physical_memory_gib": (
            round(memory_bytes / float(1024 ** 3), 2) if memory_bytes is not None else None
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _host_matches_requirement(
    host: dict[str, Any], requirement: dict[str, Any] | None
) -> tuple[bool, str | None]:
    if not requirement:
        return True, None

    required_gib = requirement.get("memory_gib_eq")
    tolerance_gib = requirement.get("tolerance_gib", 0.75)
    observed_gib = host.get("physical_memory_gib")
    if required_gib is not None:
        if observed_gib is None:
            return False, "host_physical_memory_unknown"
        if abs(float(observed_gib) - float(required_gib)) > float(tolerance_gib):
            return (
                False,
                f"host_memory_mismatch(required={required_gib}GiB±{tolerance_gib}, observed={observed_gib}GiB)",
            )
    return True, None


def _expand_template(value: str, variables: dict[str, str]) -> str:
    return Template(value).safe_substitute(variables)


def _case_command(case: dict[str, Any], host: dict[str, Any], variables: dict[str, str]) -> str | None:
    if "command" in case:
        return _expand_template(str(case["command"]), variables)
    platform_map = case.get("command_by_platform")
    if isinstance(platform_map, dict):
        command = platform_map.get(host["platform_system"])
        if command:
            return _expand_template(str(command), variables)
    return None


def _parse_kv_lines(stdout: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "--":
            if current:
                rows.append(current)
                current = {}
            continue
        match = KV_LINE_RE.match(line)
        if match:
            key, value = match.groups()
            current[key] = value
    if current:
        rows.append(current)
    return {
        "row_count": len(rows),
        "rows": rows,
    }


def _run_case(
    case: dict[str, Any],
    host: dict[str, Any],
    variables: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    requirement = case.get("requires_host")
    host_ok, host_reason = _host_matches_requirement(host, requirement)
    command = _case_command(case, host, variables)

    result: dict[str, Any] = {
        "id": case["id"],
        "description": case.get("description"),
        "category": case.get("category"),
        "honesty_notes": case.get("honesty_notes", []),
        "requires_host": requirement,
        "command": command,
        "cwd": _expand_template(case.get("cwd", "${repo_root}"), variables),
        "expected_artifacts": case.get("expected_artifacts", []),
        "status": "pending",
    }

    if not host_ok:
        result["status"] = "blocked_host_requirement"
        result["blocker"] = host_reason
        return result

    if not command:
        result["status"] = "awaiting_manual_capture"
        result["blocker"] = "no_command_for_platform"
        return result

    if dry_run:
        result["status"] = "planned"
        return result

    cwd = Path(result["cwd"])
    env = os.environ.copy()
    for key, value in variables.items():
        env[key.upper()] = value

    shell = host["platform_system"] == "windows"
    argv: str | list[str]
    if shell:
        argv = command
    else:
        argv = shlex.split(command)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        shell=shell,
    )
    duration_ms = round((time.perf_counter() - start) * 1000.0, 3)

    result.update(
        {
            "status": "captured" if completed.returncode == 0 else "failed",
            "started_at": started_at,
            "duration_ms": duration_ms,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "parsed_output": _parse_kv_lines(completed.stdout),
        }
    )
    return result


def _materialize_run(template: dict[str, Any], host: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    repo_root = Path(template.get("repo_root", ".")).resolve()
    build_dir = Path(template.get("build_dir", repo_root / "build")).resolve()
    variables = {
        "repo_root": str(repo_root),
        "build_dir": str(build_dir),
        "engine_root": str((repo_root / "engine" / "c").resolve()),
    }
    variables.update({str(k): str(v) for k, v in template.get("variables", {}).items()})
    for _ in range(3):
        variables = {key: _expand_template(value, variables) for key, value in variables.items()}

    return {
        "schema": SCHEMA_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite_id": template["suite_id"],
        "suite_description": template.get("suite_description"),
        "template_path": template.get("_template_path"),
        "dry_run": dry_run,
        "host": host,
        "suite_requires_host": template.get("suite_requires_host"),
        "variables": variables,
        "cases": [
            _run_case(case, host, variables, dry_run=dry_run) for case in template.get("cases", [])
        ],
    }


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA_ID:
        errors.append(f"schema must be {SCHEMA_ID}")
    if not payload.get("suite_id"):
        errors.append("suite_id is required")
    if "host" not in payload or not isinstance(payload["host"], dict):
        errors.append("host block is required")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty array")
        return errors

    allowed_status = {
        "pending",
        "planned",
        "captured",
        "failed",
        "blocked_host_requirement",
        "awaiting_manual_capture",
    }
    for index, case in enumerate(cases):
        label = f"cases[{index}]"
        if not case.get("id"):
            errors.append(f"{label}.id is required")
        status = case.get("status")
        if status not in allowed_status:
            errors.append(f"{label}.status invalid: {status}")
        if status in {"captured", "failed"}:
            if "exit_code" not in case:
                errors.append(f"{label}.exit_code required when status={status}")
            if "duration_ms" not in case:
                errors.append(f"{label}.duration_ms required when status={status}")
            if "stdout" not in case or "stderr" not in case:
                errors.append(f"{label}.stdout and .stderr required when status={status}")
        if status == "blocked_host_requirement" and not case.get("blocker"):
            errors.append(f"{label}.blocker required when host requirement blocks execution")
    return errors


def cmd_probe_host(args: argparse.Namespace) -> int:
    payload = probe_host()
    if args.output:
        _write_json(Path(args.output), payload)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    template_path = Path(args.template).resolve()
    template = _load_json(template_path)
    template["_template_path"] = str(template_path)
    payload = _materialize_run(template, probe_host(), dry_run=args.dry_run)
    errors = validate_payload(payload)
    if errors:
        for error in errors:
            print(f"validation_error={error}", file=sys.stderr)
        return 2
    if args.output:
        _write_json(Path(args.output), payload)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    payload = _load_json(Path(args.input).resolve())
    errors = validate_payload(payload)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2
    print("validation=ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible benchmark harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe-host", help="print host facts without guessing")
    probe.add_argument("--output", help="optional JSON output path")
    probe.set_defaults(func=cmd_probe_host)

    run = subparsers.add_parser("run", help="materialize or execute a benchmark template")
    run.add_argument("--template", required=True, help="template JSON path")
    run.add_argument("--output", help="captured JSON output path")
    run.add_argument("--dry-run", action="store_true", help="do not execute commands")
    run.set_defaults(func=cmd_run)

    validate = subparsers.add_parser("validate", help="validate a captured JSON file")
    validate.add_argument("--input", required=True, help="captured JSON path")
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
