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

  const version = runCli(['--version']);
  assert.equal(version.status, 0, version.stderr);
  assert.equal(version.stdout.trim(), packageJson.version);
});

test('doctor fails clearly when python is unavailable', () => {
  const result = runCli(['doctor', '--help'], { env: { PATH: '' } });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Python 3 was not found/);
});

test('chat fails clearly when npm is unavailable', () => {
  const result = runCli(['chat', '--build'], { env: { PATH: '' } });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /npm was not found/);
});

test('serve --native fails clearly when the native binary is unavailable', () => {
  const result = runCli(['serve', '--native'], { env: { PATH: '' } });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /serve --native requires the built native binary/);
});

test('unknown commands are rejected', () => {
  const result = runCli(['produto']);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /unknown command/);
});
