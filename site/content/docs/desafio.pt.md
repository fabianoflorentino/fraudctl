---
title: "Desafio"
description: "Escopo, contrato funcional, restrições de infraestrutura e critérios de avaliação da Rinha de Backend 2026."
weight: 10
---

## Contexto

A Rinha de Backend 2026 propôs a construção de um módulo de detecção de fraude para transações de cartão com base em busca vetorial. O sistema deve receber os dados de uma transação, compará-la com uma base histórica de 3 milhões de registros e retornar uma decisão de aprovação acompanhada de um score de risco.

O módulo é consumido por um sistema de autorização de cartão. O escopo do desafio cobre exclusivamente a API de detecção, não o fluxo completo de autorização.

## Contrato funcional

| Endpoint | Método | Resposta |
|---|---|---|
| `/ready` | `GET` | `2xx` quando a API está pronta |
| `/fraud-score` | `POST` | `200 OK` com `{approved, fraud_score}` |

O payload contém dados da transação, do cliente, do comerciante, do terminal e, opcionalmente, da transação anterior. A especificação completa está em [API.md](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/API.md).

## Regra de decisão

1. Transformar o payload em um vetor de **14 dimensões**.
2. Buscar os **5 vetores mais próximos** na base de referência.
3. Calcular `fraud_score` como a fração de vizinhos classificados como fraude.
4. Aprovar a transação quando `fraud_score < 0,6`.

O score assume apenas seis valores — `0.0`, `0.2`, `0.4`, `0.6`, `0.8` e `1.0` — o que permite pré-computar as respostas e reduzir o custo de formatação no caminho crítico.

## Restrições de infraestrutura

- Pelo menos **um balanceador de carga** e **duas instâncias da API**, com distribuição round-robin.
- O balanceador **não pode** aplicar lógica de detecção.
- `docker-compose.yml` com imagens públicas compatíveis com `linux-amd64`.
- Rede **bridge**; modos `host` e `privileged` são proibidos.
- Soma dos limites de todos os serviços: **1 CPU e 350 MB**.
- Porta pública: **9999**.
- Proibido usar os payloads de teste como referência ou realizar lookup por identificador.

## Critérios de avaliação

O score final combina duas componentes independentes, cada uma entre −3000 e +3000:

- **Latência (`score_p99`)** — cada melhora de 10× vale +1000 pontos; satura em +3000 com p99 ≤ 1 ms; é fixada em −3000 acima de 2000 ms.
- **Detecção (`score_det`)** — considera falsos positivos, falsos negativos e erros HTTP. Acima de 15% de falhas, o score é fixado em −3000.

O fraudctl encerrou a edição com **5.935,21 pontos** e **zero falhas** em 50 mil requisições. A leitura detalhada está em [Resultados](../resultados/).
