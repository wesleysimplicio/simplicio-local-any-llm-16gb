#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const repoRoot = path.resolve(__dirname, '..');
const pkg = require(path.join(repoRoot, 'package.json'));

function exists(relPath) {
  return fs.existsSync(path.join(repoRoot, relPath));
}

function resolveNativeCli() {
  const candidates = [
    'build/apps/us4-cli.exe',
    'build/apps/us4-cli',
    'build/us4-cli.exe',
    'build/us4-cli',
    'build/Release/us4-cli.exe',
  ];
  return candidates
    .map((candidate) => path.join(repoRoot, candidate))
    .find((candidate) => fs.existsSync(candidate)) || null;
}

function resolvePython() {
  if (process.env.US4_PYTHON) {
    return { cmd: process.env.US4_PYTHON, args: [] };
  }
  const candidates = [
    { cmd: 'python3', args: [] },
    { cmd: 'python', args: [] },
    { cmd: 'py', args: ['-3'] },
  ];
  for (const candidate of candidates) {
    const probe = spawnSync(candidate.cmd, [...candidate.args, '--version'], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: 'ignore',
    });
    if (probe.status === 0) return candidate;
  }
  return null;
}

function resolveNpm() {
  const candidates = process.platform === 'win32'
    ? ['npm.cmd', 'npm']
    : ['npm'];
  for (const candidate of candidates) {
    const probe = spawnSync(candidate, ['--version'], {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: 'ignore',
    });
    if (probe.status === 0) return candidate;
  }
  return null;
}

function run(command, args, extraEnv = undefined) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: 'inherit',
    env: extraEnv ? { ...process.env, ...extraEnv } : process.env,
  });
  process.exit(result.status ?? 1);
}

function printHelp() {
  console.log(`us4-cli product wrapper

Usage:
  us4-cli probe [--json]
  us4-cli list-models [--json]
  us4-cli run ...
  us4-cli serve [--native] [...]
  us4-cli doctor [...]
  us4-cli plan [...]
  us4-cli convert [...]
  us4-cli chat [--build|--preview] [...]

Product commands:
  probe        delegate to the native runtime probe
  list-models  delegate to the native adapter registry
  run          delegate to the native single-shot runtime
  serve        delegate to the native runtime when built; otherwise
               falls back to scripts/openai_serve.py for proxy mode
  doctor       run engine/c/doctor.py
  plan         run engine/c/resource_plan.py
  convert      run engine/c/tools/convert_fp8_to_int4.py
  chat         run apps/web-chat via npm (dev by default)

Legacy scaffold CLI remains available as:
  node bin/cli.js --help

Version:
  ${pkg.version}`);
}

function requirePython(scriptRelPath) {
  const python = resolvePython();
  if (!python) {
    console.error('us4-cli: Python 3 was not found. Set US4_PYTHON or install python3/python/py -3.');
    process.exit(1);
  }
  const scriptPath = path.join(repoRoot, scriptRelPath);
  if (!fs.existsSync(scriptPath)) {
    console.error(`us4-cli: script not found: ${scriptRelPath}`);
    process.exit(1);
  }
  return { python, scriptPath };
}

function runPythonScript(scriptRelPath, rest) {
  const { python, scriptPath } = requirePython(scriptRelPath);
  run(python.cmd, [...python.args, scriptPath, ...rest]);
}

function runChat(rest) {
  const npm = resolveNpm();
  if (!npm) {
    console.error('us4-cli chat: npm was not found on PATH.');
    process.exit(1);
  }
  if (!exists('apps/web-chat/package.json')) {
    console.error('us4-cli chat: apps/web-chat/package.json not found.');
    process.exit(1);
  }

  let script = 'dev';
  const passthrough = [];
  for (const arg of rest) {
    if (arg === '--build') {
      script = 'build';
      continue;
    }
    if (arg === '--preview') {
      script = 'preview';
      continue;
    }
    passthrough.push(arg);
  }

  const npmArgs = ['--prefix', path.join(repoRoot, 'apps', 'web-chat'), 'run', script];
  if (passthrough.length) {
    npmArgs.push('--', ...passthrough);
  }
  run(npm, npmArgs);
}

function runNative(rest) {
  const nativeCli = resolveNativeCli();
  if (!nativeCli) {
    console.error('us4-cli: native runtime binary not found. Build with: cmake --build build --target us4-cli');
    process.exit(1);
  }
  run(nativeCli, rest);
}

function runServe(rest) {
  const nativeCli = resolveNativeCli();
  if (nativeCli) {
    run(nativeCli, ['serve', ...rest]);
  }
  if (rest.includes('--native')) {
    console.error('us4-cli serve --native requires the built native binary. Build with: cmake --build build --target us4-cli');
    process.exit(1);
  }
  runPythonScript('scripts/openai_serve.py', rest);
}

const args = process.argv.slice(2);

if (args.length === 0 || args[0] === '--help' || args[0] === '-h' || args[0] === 'help') {
  printHelp();
  process.exit(0);
}

if (args[0] === '--version' || args[0] === '-v') {
  console.log(pkg.version);
  process.exit(0);
}

const [command, ...rest] = args;

switch (command) {
  case 'doctor':
    runPythonScript('engine/c/doctor.py', rest);
    break;
  case 'plan':
    runPythonScript('engine/c/resource_plan.py', rest);
    break;
  case 'convert':
    runPythonScript('engine/c/tools/convert_fp8_to_int4.py', rest);
    break;
  case 'chat':
    runChat(rest);
    break;
  case 'serve':
    runServe(rest);
    break;
  case 'probe':
    runNative(['--probe', ...rest]);
    break;
  case 'list-models':
    runNative(['list-models', ...rest]);
    break;
  case 'run':
    runNative(['run', ...rest]);
    break;
  default:
    if (command.startsWith('--')) {
      runNative(args);
      break;
    }
    console.error(`us4-cli: unknown command "${command}"`);
    printHelp();
    process.exit(1);
}
