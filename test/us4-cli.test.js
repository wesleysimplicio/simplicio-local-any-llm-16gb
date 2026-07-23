'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const cli = path.join(root, 'bin', 'us4-cli.js');
const packageJson = require(path.join(root, 'package.json'));

function runCli(args, options = {}) {
  return spawnSync(process.execPath, [cli, ...args], {
    cwd: root,
    encoding: 'utf8',
    env: { ...process.env, NO_COLOR: '1', ...(options.env || {}) },
  });
}

test('prints product help and version', () => {
  const help = runCli(['--help']);
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /us4-cli product wrapper/);
  assert.match(help.stdout, /doctor/);
  assert.match(help.stdout, /plan/);
  assert.match(help.stdout, /convert/);
  assert.match(help.stdout, /chat/);
  assert.match(help.stdout, /serve/);
  assert.match(help.stdout, /backend/);
  assert.match(help.stdout, /prototype/);

  const version = runCli(['--version']);
  assert.equal(version.status, 0, version.stderr);
  assert.equal(version.stdout.trim(), packageJson.version);
});

test('local inference status is read-only and exposes the stable pause reason', () => {
  const paused = runCli(['local-inference-status', '--json'], { env: { PATH: '' } });
  assert.equal(paused.status, 0, paused.stderr);
  assert.deepEqual(JSON.parse(paused.stdout), {
    local_inference: 'paused',
    reason: 'LOCAL_INFERENCE_PAUSED',
  });

  const admitted = runCli(['local-inference-status', '--json'], {
    env: {
      PATH: '',
      US4_LOCAL_INFERENCE: 'enabled',
      US4_RUNTIME_POLICY: 'admitted',
      US4_RUNTIME_LEASE: 'lease-test-only',
    },
  });
  assert.equal(admitted.status, 0, admitted.stderr);
  assert.deepEqual(JSON.parse(admitted.stdout), {
    local_inference: 'enabled',
    reason: 'runtime-policy-admitted',
  });
});

test('doctor fails clearly when python is unavailable', () => {
  const result = runCli(['doctor', '--help'], { env: { PATH: '' } });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Python 3 was not found/);
});

test('backend probe is read-only and versioned', () => {
  const result = runCli(['backend', 'probe', '--json']);
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.protocol, 'simplicio.local-inference-backend/v1');
  assert.equal(report.read_only, true);
  assert.equal(report.model_payload_read, false);
  assert.equal(report.capabilities.tool_execution, 'forbidden');
});

test('prototype doctor preserves offline and no-effect policy', () => {
  const result = runCli(['prototype', 'doctor', '--json']);
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.protocol, 'simplicio.prototype-worker/v1');
  assert.equal(report.offline, true);
  assert.equal(report.prompt_logging, false);
  assert.equal(report.effect_authority, 'none');
});

test('prototype generate escalates without a Runtime inference lease', () => {
  const result = runCli([
    'prototype', 'generate', '--artifact-type', 'plan', '--json',
  ]);
  assert.equal(result.status, 78, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.status, 'escalate_remote');
  assert.equal(report.reason, 'runtime-inference-lease-required');
});

test('chat fails clearly when npm is unavailable', () => {
  const result = runCli(['chat', '--build'], { env: { PATH: '' } });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /npm was not found/);
});

test('serve --native is paused before checking for a native binary', () => {
  const result = runCli(['serve', '--native'], { env: { PATH: '' } });
  assert.equal(result.status, 78);
  assert.match(result.stderr, /LOCAL_INFERENCE_PAUSED/);
});

test('local run and serve are paused before process, network, or model access', () => {
  for (const args of [['run', '--model', 'qwen-0.5b'], ['serve']]) {
    const result = runCli(args, { env: { PATH: '' } });
    assert.equal(result.status, 78, result.stderr);
    assert.match(result.stderr, /LOCAL_INFERENCE_PAUSED/);
    assert.match(result.stderr, /Runtime lease/);
  }
});

test('runtime admission is required as a complete policy and lease tuple', () => {
  const incomplete = runCli(['serve'], {
    env: { PATH: '', US4_LOCAL_INFERENCE: 'enabled', US4_RUNTIME_POLICY: 'admitted' },
  });
  assert.equal(incomplete.status, 78);
  assert.match(incomplete.stderr, /LOCAL_INFERENCE_PAUSED/);

  const admitted = runCli(['serve', '--native'], {
    env: {
      PATH: '',
      US4_LOCAL_INFERENCE: 'enabled',
      US4_RUNTIME_POLICY: 'admitted',
      US4_RUNTIME_LEASE: 'lease-test-only',
    },
  });
  assert.equal(admitted.status, 1);
  assert.doesNotMatch(admitted.stderr, /LOCAL_INFERENCE_PAUSED/);
  assert.match(admitted.stderr, /serve --native requires the built native binary/);
});

test('unknown commands are rejected', () => {
  const result = runCli(['produto']);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /unknown command/);
});
