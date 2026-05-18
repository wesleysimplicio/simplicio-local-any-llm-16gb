#Speculative

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

        O foco aqui ainda e corretude de superficie.Telemetria de acceptance e
            loader do draft model entram nos proximos slices do Sprint 10.
