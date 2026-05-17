# Evaluation

## Objetivo

Este documento resume como o projeto é avaliado e como isso se conecta com a implementação atual.

## O que a Rinha Mede

Para cada request do conjunto de teste:

- `TP`: fraude negada corretamente
- `TN`: legítima aprovada corretamente
- `FP`: legítima negada incorretamente
- `FN`: fraude aprovada incorretamente
- `HTTP error`: resposta diferente de `200`

## Fórmula de Score

### Latência

```text
Se p99 > 2000ms:
  score_p99 = -3000
Senão:
  score_p99 = 1000 * log10(1000 / max(p99, 1))
```

### Qualidade de detecção

```text
E = 1*FP + 3*FN + 5*HTTP_ERRORS
epsilon = E / N
failure_rate = (FP + FN + HTTP_ERRORS) / N

Se failure_rate > 15%:
  score_det = -3000
Senão:
  score_det = 1000 * log10(1 / max(epsilon, 0.001)) - 300 * log10(1 + E)
```

### Score final

```text
final_score = score_p99 + score_det
```

## Como a Implementação Atual Ajuda no Score

### Para p99

- índice IVF pronto em tempo de build
- parser JSON rápido no hot path
- `GOMAXPROCS=1` por instância
- LB custom com passagem de FD, evitando proxy HTTP tradicional no caminho principal
- respostas pré-computadas

### Para qualidade

- KNN com `K=5`
- zona ambígua tratada com second pass no IVF
- bbox pruning para gastar CPU onde importa
- fallback JSON em vez de erro HTTP quando o parse falha

## Compliance Atual com o Desafio

O projeto atual está alinhado com as regras usuais da Rinha neste ponto:

- sem cache por payload de teste
- sem lookup por ID de request
- pré-processamento dos arquivos de recursos em tempo de build
- materialização do índice `ivf.bin` antes do runtime

## Observações Sobre o Estado do Repositório

Alguns documentos antigos descreviam:

- `nginx` ou `haproxy` como caminho principal
- score histories antigas
- valores de `nprobe` e `quickProbe` que não batem com o código atual

Este documento foi ajustado para priorizar a implementação atual e não histórico experimental antigo.

## Ferramentas de Avaliação no Repositório

- `make k6-smoke`
- `make k6-full`
- `make k6-results`
- `cmd/debug` para inspeções HTTP simples
- `cmd/analyze` para exploração de bandas de prefilter em dataset rotulado

## Nota Sobre Prefilter

O repositório contém artefatos e carregamento de `model.bin` e `gbdt.bin`, mas o handler principal atual não usa esse prefilter no fluxo padrão da API. Portanto, a avaliação operacional atual do serviço é baseada no KNN IVF.
