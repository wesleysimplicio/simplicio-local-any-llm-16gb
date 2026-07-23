#!/usr/bin/env bash
# Hook pre-commit do Claude Code.
# Roda a suite de testes em modo silencioso e BLOQUEIA o commit se vermelho.
# Mensagens em pt-BR para feedback rápido. CI roda o set completo (lint + e2e);
# aqui o foco é evitar commit obviamente quebrado.

set -euo pipefail

echo "[pre-commit] Rodando testes locais antes do commit..."

# Garante que existe package.json antes de tentar npm test.
if [[ ! -f "package.json" ]]; then
  echo "[pre-commit] package.json não encontrado. Pulando testes."
  exit 0
fi

# Roda npm test silenciosamente. Se falhar, bloqueia.
if ! npm test --silent; then
  echo ""
  echo "[pre-commit] FALHOU: testes vermelhos. Commit bloqueado."
  echo "[pre-commit] Corrija os testes antes de commitar."
  echo "[pre-commit] Para depurar: rode 'npm test' direto e leia o output."
  exit 1
fi

if ! command -v make >/dev/null 2>&1; then
  echo "[pre-commit] FALHOU: 'make' é obrigatório para validar o motor colibri."
  exit 1
fi

if ! make -C engine/c test; then
  echo ""
  echo "[pre-commit] FALHOU: suíte local do motor colibri vermelha."
  echo "[pre-commit] Corrija antes de commitar: make -C engine/c test"
  exit 1
fi

echo "[pre-commit] Testes verdes. Seguindo com o commit."
exit 0
