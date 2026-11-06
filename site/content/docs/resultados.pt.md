---
title: "Resultados"
description: "Números oficiais da Rinha de Backend 2026, composição do score e leitura executiva do resultado."
weight: 60
---

## Resultado oficial

Os números abaixo são provenientes de `final-results-2026.json`, publicado em [rinhadebackend.com.br](https://rinhadebackend.com.br/). A submissão avaliada foi o commit `e90aaa4`, imagem `fraudctl:v1.0.114`.

| Métrica | Valor |
|---|---:|
| Colocação | **19º / 322** |
| Score final | **5.935,21 / 6.000** |
| p99 | **1,1609 ms** |
| Requisições avaliadas | 50.000 |
| Casos de borda | 5.605 |
| Falsos positivos | **0** |
| Falsos negativos | **0** |
| Erros HTTP | **0** |

## Composição do score

| Componente | Pontos |
|---|---:|
| Latência (p99) | 2.935,21 / 3.000 |
| Detecção | 3.000 / 3.000 |
| **Total** | **5.935,21 / 6.000** |

A componente de detecção atingiu a pontuação máxima porque não houve erro de decisão nem erro HTTP. A latência ficou ligeiramente acima de 1 ms, faixa em que o score satura em 3.000.

## Leitura executiva

- **Zero erros de decisão** em 50 mil requisições, incluindo 5.605 casos de borda.
- **p99 de 1,16 ms** com 1 CPU e 350 MB no total.
- **19ª colocação** entre 322 submissões válidas, à frente de soluções com mais recursos.

A maior parte das submissões com pontuação superior saturou a componente de latência (p99 ≤ 1 ms). O fraudctl atingiu 1,16 ms — próximo o suficiente para garantir 2.935 dos 3.000 pontos, com margem para evolução.

## Próximos passos

Com mais tempo de desenvolvimento, as frentes naturais seriam:

- reduzir o p99 para abaixo de 1 ms;
- medir o impacto real de um prefilter calibrado sem sacrificar exatidão;
- explorar melhor o paralelismo entre as duas instâncias;
- manter o índice ainda mais próximo da cache L3.
