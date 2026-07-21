'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const script = path.join(root, 'scripts', 'openai_serve.py');

test('direct serve script is paused before dependency import, bind, or upstream contact', () => {
  const result = spawnSync('python3', [script], {
    cwd: root,
    encoding: 'utf8',
    env: { ...process.env, US4_SERVE_PORT: '0' },
  });
  assert.equal(result.status, 78, result.stderr);
  assert.match(result.stderr, /LOCAL_INFERENCE_PAUSED/);
});

test('policy helper needs Runtime admission and a non-empty lease', () => {
  const probe = [
    'import os,sys;',
    `sys.path.insert(0, ${JSON.stringify(path.join(root, 'scripts'))});`,
    'import openai_serve;',
    'print(openai_serve._runtime_admission()[0])',
  ].join('');
  const denied = spawnSync('python3', ['-c', probe], { encoding: 'utf8' });
  assert.equal(denied.status, 0, denied.stderr);
  assert.equal(denied.stdout.trim(), 'False');

  const admitted = spawnSync('python3', ['-c', probe], {
    encoding: 'utf8',
    env: {
      ...process.env,
      US4_LOCAL_INFERENCE: 'enabled',
      US4_RUNTIME_POLICY: 'admitted',
      US4_RUNTIME_LEASE: 'lease-test-only',
    },
  });
  assert.equal(admitted.status, 0, admitted.stderr);
  assert.equal(admitted.stdout.trim(), 'True');
});
