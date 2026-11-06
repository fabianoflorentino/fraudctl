---
title: "Documentação"
description: "Visão técnica e executiva do fraudctl: contexto, arquitetura, desempenho e decisões de engenharia da 19ª colocação na Rinha de Backend 2026."
weight: 1
---

## Resumo executivo

O **fraudctl** é um módulo de detecção de fraude em transações de cartão, desenvolvido para a Rinha de Backend 2026. O desafio exigia transformar cada transação em um vetor de 14 dimensões, localizar os cinco registros históricos mais semelhantes em uma base de 3 milhões de referências e devolver uma decisão de aprovação em tempo real, com pelo menos um balanceador de carga e duas instâncias da API.

A solução foi implementada em Go, com índice de busca exato construído em tempo de build, e executada sob um orçamento total de **1 CPU e 350 MB** de memória.

**Resultado oficial:** 19º lugar entre 322 submissões, com score de **5.935,21 de 6.000 pontos**, p99 de **1,16 ms** e nenhum falso positivo, falso negativo ou erro HTTP em 50 mil requisições avaliadas.

## Objetivo desta documentação

Esta seção apresenta as decisões técnicas e os trade-offs que sustentam o resultado, em linguagem acessível a líderes de produto, gestores de engenharia e avaliadores técnicos. Cada página combina contexto de negócio, critérios de decisão e evidências de implementação.

## Estrutura

- **Desafio** — escopo, contrato funcional, restrições e critérios de avaliação.
- **Arquitetura** — topologia, componentes e justificativa das escolhas estruturais.
- **Detecção** — modelo de decisão, dimensões e processo de busca.
- **Performance** — metas, resultados e otimizações com impacto mensurado.
- **Go idiomático** — padrões de engenharia, qualidade e operação.
- **Resultados** — números oficiais e leitura do score.
- **Lições** — decisões mantidas, descartadas e recomendações.

{{< callout >}}
**Para gestão e produto:** os números oficiais, o impacto e a composição do score estão consolidados em [Resultados](resultados/). As demais páginas aprofundam o "como" e o "porquê" das decisões.
{{< /callout >}}
