---
title: "Desafio"
description: "O contrato da Rinha de Backend 2026, as restrições de infraestrutura e como o score é calculado."
weight: 10
---

A Rinha de Backend 2026 propôs um módulo de **detecção de fraude em transações de cartão usando busca vetorial**. Para cada transação, a API precisa transformar o payload em um vetor, encontrar os 5 vizinhos mais próximos em uma base de 3 milhões de referências e devolver uma decisão.

## Contrato

| Endpoint | Método | Resposta |
|---|---|---|
| `/ready` | `GET` | `2xx` quando a API está pronta |
| `/fraud-score` | `POST` | `200 OK` com `{approved, fraud_score}` |

O payload contém a transação, o cliente, o comerciante, o terminal e, opcionalmente, a última transação. A especificação completa está em [API.md](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/API.md).

## Regra de decisão

1. Transformar o payload em um vetor de **14 dimensões**.
2. Buscar os **5 vetores mais próximos** no dataset de referência.
3. Calcular `fraud_score = fraudes_entre_os_5 / 5`.
4. Aprovar quando `fraud_score < 0,6`.

O `fraud_score` assume apenas seis valores: `0.0`, `0.2`, `0.4`, `0.6`, `0.8` e `1.0`. Isso permite pré-computar as respostas.

## Restrições

- Pelo menos **um load balancer** e **duas instâncias da API**, em round-robin.
- O load balancer **não pode** aplicar lógica de detecção.
- `docker-compose.yml` com imagens públicas `linux-amd64`.
- Rede **bridge**; `host` e `privileged` são proibidos.
- Soma dos limites de todos os serviços: **1 CPU e 350 MB**.
- Porta pública: **9999**.
- Proibido usar os payloads de teste como referência ou fazer lookup por ID.

## Como o score é calculado

O score final soma duas componentes independentes, cada uma entre −3000 e +3000:

- **Latência (`score_p99`)** — melhora de 10× vale +1000 pontos; satura em +3000 com p99 ≤ 1 ms; vira −3000 se passar de 2000 ms.
- **Detecção (`score_det`)** — combina falsos positivos, falsos negativos e erros HTTP. Acima de 15% de falhas, o score é fixado em −3000.

O `fraudctl` terminou com **5935,21 pontos** e **0 falhas** em 50 mil requisições. Os detalhes estão em [Resultados](../resultados/).
