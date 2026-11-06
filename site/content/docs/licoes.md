---
title: "Lições"
description: "O que foi testado, o que ficou, o que foi descartado e o que faria diferente."
weight: 70
---

A versão final é o resultado de várias abordagens testadas e medidas. Algumas ficaram; outras foram removidas sem dó.

## O que ficou

**Índice exato com KD-tree particionada.** A combinação de partição por features, KD-tree por partição, poda por bounding box e kernel AVX2 entregou exatidão e latência ao mesmo tempo.

**Load balancer com SCM_RIGHTS.** Remover o proxy HTTP entre o LB e as APIs simplificou o caminho crítico e reduziu overhead sem violar a regra de que o LB não aplica lógica de detecção.

**Servidor HTTP sob medida.** Para dois endpoints e um contrato fixo, um parser orientado a bytes foi mais previsível do que `net/http` ou `fasthttp`.

**Build-time index.** Gerar o índice no build da imagem eliminou trabalho de startup e manteve o runtime focado em responder.

## O que foi descartado

**IVF aproximado.** Excelente no microbenchmark, mas aproximado. A busca exata evitava a classe de erro mais cara do score.

**Prefilter GBDT/RF.** Modelos de árvore foram treinados e avaliados, mas a calibração para garantir zero FP/FN não compensou a complexidade no caminho final.

**Fallbacks não usados.** Depois do evento, o repositório passou por uma limpeza: código morto, fallbacks não exercitados e artefatos de modelo saíram do caminho principal.

## O que faria diferente

- **Medir antes de otimizar.** Várias otimizações só fizeram sentido depois de um perfil de CPU mostrar onde o tempo realmente estava.
- **Tratar o p99 como alvo de primeira classe.** A diferença entre 1,16 ms e 1,0 ms vale 65 pontos.
- **Documentar as decisões durante o processo.** A documentação posterior é boa, mas perde o contexto de cada trade-off.
- **Simplificar mais cedo.** O código de experimentos tende a sobreviver ao experimento; remover cedo reduz o custo de manutenção.

## O principal aprendizado

Performance em Go não é sobre micro-otimizar tudo. É sobre **encontrar o caminho quente**, remover alocação, escolher a estrutura de dados certa e isolar o que precisa de assembly. O resto deve continuar simples, testável e idiomático.
