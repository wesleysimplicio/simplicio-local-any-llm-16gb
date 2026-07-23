# Speculative

Home do contrato de speculative decoding.

        ##PEagleDecoder

`PEagleDecoder` abre o caminho draft /
        verify com tres garantias nesta fase :

    -draft clampado por janela configuravel -
    prefixo aceito + fallback explicito no primeiro mismatch -
    equivalencia observavel com o caminho autoritativo nao -
    speculative

        ##Eagle3Decoder

`Eagle3Decoder` amplia isso para verify por arvore :

    -clamp de breadth /
    depth -
    escolha do ramo com maior prefixo compartilhado -
    fallback explicito para manter equivalencia com o caminho autoritativo

## Política adaptativa de 16 GB

`Make16GbAdaptiveSpeculativeConfig()` limita o lookahead a dois tokens. O MTP
só é liberado depois do warm-up quando:

- acceptance acumulada é pelo menos 75%;
- hit-rate do cache de experts é pelo menos 60%;
- o lookahead saiu do mínimo conservador.

As flags manuais do motor C continuam tendo precedência. Os limiares são
defaults seguros de fixture e precisam ser recalibrados com o benchmark de
hardware real.

## Commit lossless, limites e cancelamento

`LosslessSpeculativeSession` envolve o verify do `PEagleDecoder` e oferece:

- limite de draft, rounds e tokens comitados;
- cancelamento cooperativo por `std::stop_token`;
- métricas de attempts, accepted/rejected, commits, cancelamentos e limites;
- commit somente do prefixo validado contra a sequência autoritativa e do
  fallback autoritativo no primeiro mismatch.

Um cancelamento observado antes do round retorna zero tokens. Limites podem
truncar um prefixo já verificado, mas nunca tornam um token de draft
autoritativo.
