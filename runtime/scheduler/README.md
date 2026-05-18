# Runtime Scheduler

O scheduler abre a historia multi-sessao do runtime.

## ContinuousBatcher

`ContinuousBatcher` distribui um orcamento fixo de tokens por batch usando
rodadas deterministicas por `arrivalOrder`.

- `fairnessWeight` permite dar mais de um token por rodada a uma sessao.
- sessoes com `pendingTokens == 0` ficam fora do ciclo.
- `singleSessionPassthrough` deixa explicito quando nao houve contencao.

Nesta fase, o contrato e de superficie: ele protege fairness e regressao de
single-session antes de abrirmos `SessionPool` e speculative decoding.
