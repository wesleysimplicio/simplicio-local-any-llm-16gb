'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.resolve(__dirname, '..');
const read = (relative) => fs.readFileSync(path.join(root, relative), 'utf8');

test('active CI workflow runs on main pushes and pull requests', () => {
  const workflow = read('.github/workflows/ci.yml');
  assert.match(workflow, /on:\n  push:\n    branches: \[main\]/);
  assert.match(workflow, /pull_request:\n    branches: \[main\]/);
  assert.match(workflow, /Enforce starter line coverage \(>=80%\)/);
  assert.doesNotMatch(workflow, /#\s+push:/);
  assert.doesNotMatch(workflow, /#\s+pull_request:/);
});

test('active DoD workflow is attached to pull requests', () => {
  const workflow = read('.github/workflows/dod.yml');
  assert.match(workflow, /on:\n  pull_request:/);
  assert.match(workflow, /branches: \[main\]/);
  assert.match(workflow, /Check starter coverage threshold \(>=80%\)/);
});

test('scaffold self-check is scoped to this repository', () => {
  const workflow = read('.github/workflows/scaffold-self-check.yml');
  assert.doesNotMatch(workflow, /wesleysimplicio\/agentic-starter/);
  assert.match(workflow, /wesleysimplicio\/simplicio-local/);
});
