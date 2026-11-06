---
title: "Lições"
description: "Decisões mantidas, decisões descartadas e recomendações para projetos futuros."
weight: 70
---

## Decisões mantidas

**Índice exato com KD-tree particionada.** A combinação de partição por features, KD-tree por partição, poda por bounding box e kernel AVX2 permitiu conciliar exatidão e latência.

**Balanceador com passagem de descritor.** A remoção do proxy HTTP entre o balanceador e as APIs simplificou o caminho crítico e reduziu overhead, sem violar a regra de que o balanceador não aplica lógica de detecção.

**Servidor HTTP dedicado.** Para dois endpoints e um contrato fixo, um parser orientado a bytes mostrou-se mais previsível do que `net/http` ou `fasthttp`.

**Índice gerado em tempo de build.** A geração do índice durante o build da imagem eliminou trabalho de startup e concentrou o runtime na resposta às requisições.

## Decisões descartadas

**IVF aproximado.** Apresentou melhor desempenho em microbenchmark, mas era aproximado. A busca exata evitava a classe de erro de maior impacto no score.

**Prefilter GBDT/RF.** Modelos de árvore foram treinados e avaliados, mas a calibração para garantir zero FP/FN não justificou a complexidade no caminho final.

**Fallbacks não utilizados.** Após o evento, o repositório passou por uma limpeza: código morto, fallbacks não exercitados e artefatos de modelo foram retirados do caminho principal.

## Recomendações

- **Medir antes de otimizar.** Diversas otimizações só fizeram sentido após um perfil de CPU indicar onde o tempo estava concentrado.
- **Tratar o p99 como alvo de primeira classe.** A diferença entre 1,16 ms e 1,0 ms representa 65 pontos no score.
- **Documentar decisões durante o processo.** A documentação posterior é útil, mas perde parte do contexto de cada trade-off.
- **Simplificar mais cedo.** Código de experimentação tende a sobreviver ao experimento; a remoção precoce reduz o custo de manutenção.

## Conclusão

Performance em Go não depende de micro-otimizar todo o código. Depende de **identificar o caminho crítico**, eliminar alocações, escolher as estruturas de dados adequadas e isolar o que realmente exige assembly. O restante deve permanecer simples, testável e idiomático.
