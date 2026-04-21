# Plano de Projeto — fraudctl (Golang)

## Visão Geral

API de detecção de fraudes em transações de cartão via busca vetorial KNN, com latência alvo p99 ≤ 10ms, rodando em 1 CPU / 350 MB RAM total.

Endpoints:

- `GET /ready` — health check
- `POST /fraud-score` — recebe transação, retorna `{ approved, fraud_score }`

---

## Estrutura de Diretórios

```bash
fraudctl/
├── cmd/
│   └── api/
│       └── main.go              # entrypoint, carrega dataset, inicia servidor
├── internal/
│   ├── handler/
│   │   ├── ready.go             # GET /ready
│   │   └── fraud_score.go       # POST /fraud-score
│   ├── vectorizer/
│   │   └── vectorizer.go        # normalização → vetor 14D
│   ├── knn/
│   │   └── knn.go               # busca KNN (euclidiana, brute-force ou ANN)
│   └── model/
│       ├── request.go           # structs de request/response HTTP
│       └── reference.go         # struct do dataset de referência
├── resources/
│   ├── references.json.gz       # 100k vetores rotulados
│   ├── example-payloads.json    # 40 payloads de exemplo para testes manuais
│   ├── example-references.json  # ~100 vetores rotulados para testes unitários
│   ├── mcc_risk.json            # risco por MCC
│   └── normalization.json       # constantes de normalização
├── test/
│   ├── test.js                  # script k6 (load test + scoring)
│   └── test-data.json           # 14.500 entradas com vetores e respostas esperadas
├── visualization/
│   ├── generate.sh              # runner Nix para o script Python
│   └── visualize_14d.py         # gera gráficos radar das 14 dimensões
├── docker-compose.yml
├── Dockerfile
└── go.mod
```

---

## Fases de Implementação

### Fase 1 — Estrutura Base

- [x] `go mod init` + criação da estrutura de pastas
- [x] Structs de request/response (`model/request.go`)
- [x] Struct de referência do dataset (`model/reference.go`)
- [x] Servidor HTTP mínimo (`net/http`) na porta 9999
- [x] `GET /ready` retornando HTTP 200

### Fase 2 — Carregamento do Dataset

- [x] Leitura e descompressão de `references.json.gz` em memória no startup
- [x] Parse de `mcc_risk.json` e `normalization.json`
- [x] Dataset armazenado como `[][]float64` para eficiência de memória
- [x] Labels armazenadas separadamente como `[]bool` (fraud=true)

### Fase 3 — Vetorização

- [x] Implementar as 14 dimensões conforme `DETECTION_RULES.md`
- [x] Função `clamp(x) float64` restringindo valores a `[0.0, 1.0]`
- [x] Tratamento de `last_transaction: null` (dimensões 5 e 6 = -1)
- [x] Cálculo de hora UTC e dia da semana (seg=0, dom=6)
- [x] Lookup de `mcc_risk` com default `0.5` para MCC desconhecido

#### Tabela das 14 Dimensões

| Idx | Dimensão | Fórmula |
| ----- | ---------- | --------- |
| 0 | `amount` | `clamp(transaction.amount / max_amount)` |
| 1 | `installments` | `clamp(transaction.installments / max_installments)` |
| 2 | `amount_vs_avg` | `clamp((transaction.amount / customer.avg_amount) / amount_vs_avg_ratio)` |
| 3 | `hour_of_day` | `hour(transaction.requested_at) / 23` |
| 4 | `day_of_week` | `day_of_week(transaction.requested_at) / 6` |
| 5 | `minutes_since_last_tx` | `clamp(minutes / max_minutes)` ou `-1` se null |
| 6 | `km_from_last_tx` | `clamp(km / max_km)` ou `-1` se null |
| 7 | `km_from_home` | `clamp(terminal.km_from_home / max_km)` |
| 8 | `tx_count_24h` | `clamp(customer.tx_count_24h / max_tx_count_24h)` |
| 9 | `is_online` | `1` se online, `0` caso contrário |
| 10 | `card_present` | `1` se cartão presente, `0` caso contrário |
| 11 | `unknown_merchant` | `1` se merchant desconhecido, `0` se conhecido |
| 12 | `mcc_risk` | `mcc_risk.json[merchant.mcc]` (default `0.5`) |
| 13 | `merchant_avg_amount` | `clamp(merchant.avg_amount / max_merchant_avg_amount)` |

### Fase 4 — Busca KNN

- [x] Brute-force KNN: distância euclidiana sobre os 100k vetores
- [x] Paralelizar com goroutines + `sync` para maximizar uso de CPU
- [x] Manter apenas top-5 (sort parcial)
- [x] Votação: `fraud_score = fraud_count / 5`
- [x] Threshold: `fraud_score >= 0.6` → `approved: false`

### Fase 5 — Handler POST /fraud-score

- [ ] Parse do JSON de entrada
- [ ] Pipeline: vectorizer → knn → montar resposta
- [ ] Fallback em caso de erro: `{ approved: true, fraud_score: 0.0 }` (evita penalidade -5)
- [ ] Pool de objetos (`sync.Pool`) para reduzir alocações no hot path

### Fase 6 — Otimizações de Performance

- [x] Layout contíguo de memória para os vetores (cache-friendly)
- [x] Benchmark com `go test -bench` para medir latência por camada
- [x] Avaliar `fasthttp` vs `net/http` padrão via benchmark (não vale a pena - resultado: manter net/http)
- [ ] Avaliar HNSW se brute-force não atingir p99 ≤ 10ms

### Fase 7 — Docker & Compose

- [ ] `Dockerfile` multi-stage (build → runtime scratch/alpine)
- [ ] Copiar `resources/` na imagem final
- [ ] `docker-compose.yml`: nginx (round-robin) + 2 instâncias API
- [ ] Limites: 1 CPU total, 350 MB RAM total entre todos os serviços
- [ ] Porta 9999 exposta no load balancer

### Fase 8 — Testes e Validação

- [x] Testes unitários do vectorizer (mock data)
- [x] Testes unitários do dataset loader
- [ ] Validar os 4 exemplos da documentação (scores esperados: 0.0, 1.0, 0.4, 1.0)
- [ ] Validação de acurácia offline contra `test/test-data.json` (14.500 entradas, 33% fraude)
- [ ] Teste de carga com k6 (`test/test.js`): rampa de 1 → 650 RPS em 60s, max 150 VUs
  - Dataset contém 4.812 fraudes, 9.688 legítimas e 157 edge cases
  - Cada entrada já tem o vetor pré-computado e a resposta esperada — útil para CI
- [ ] Teste de carga com k6 (`test/test.js`): rampa de 1 → 650 RPS em 60s, max 150 VUs
- [ ] Ajuste fino de goroutines/workers conforme CPU disponível no container

### Fase 9 — Visualização e Análise (opcional)

- [ ] Executar `visualization/generate.sh` para gerar gráficos radar das 14 dimensões
- [ ] Analisar `fraud_14d_visualization.png` (perfil médio fraude vs. legítima) para identificar dimensões mais discriminativas
- [ ] Usar os insights para priorizar dimensões no KNN ou ajustar pesos se necessário

Dimensões com maior separação entre fraude e legítima (observadas nos gráficos):

- `amount`, `amount_vs_avg`, `km_from_home`, `mcc_risk`, `unknown_merchant`

---

## Decisões Técnicas

| Decisão | Escolha | Justificativa |
| --------- | --------- | --------------- |
| HTTP server | `net/http` ou `fasthttp` | `fasthttp` tem menor latência e menos alocações |
| Busca vetorial | Brute-force paralelizado | 100k × 14D é ~5.6M ops, viável em <1ms com goroutines |
| Tipo numérico | `float32` | Metade da memória (~5.6 MB), mais rápido no pipeline |
| Dataset em memória | `[][14]float32` contíguo | Cache-friendly, sem overhead de GC |
| Load balancer | nginx (round-robin) | Simples, leve, sem lógica de negócio |
| Fallback de erro | `approved: true, score: 0.0` | Evita penalidade de -5 por erro HTTP |

---

## Restrições de Recursos

```bash
Load Balancer (nginx):  ~0.10 CPU  ~30 MB RAM
API Instance 1:         ~0.45 CPU  ~150 MB RAM
API Instance 2:         ~0.45 CPU  ~150 MB RAM
                        ──────────────────────
Total:                  ~1.00 CPU  ~330 MB RAM  ✓
```

O dataset em memória ocupa ~5.6 MB por instância com `float32`, bem dentro do orçamento de 150 MB por instância.

---

## Perfil do Dataset de Teste

O arquivo `test/test-data.json` contém o conjunto de avaliação completo:

| Métrica | Valor |
| --------- | ------- |
| Total de entradas | 14.500 |
| Transações fraudulentas | 4.812 (33,2%) |
| Transações legítimas | 9.688 (66,8%) |
| Edge cases | 157 (1,1%) |

Cada entrada já inclui o vetor 14D pré-computado e a resposta esperada (`approved` + `fraud_score`), o que permite criar um teste de acurácia offline sem precisar subir o servidor.

O script `test/test.js` (k6) implementa o perfil de carga:

| Estágio | Duração | RPS alvo |
| --------- | --------- | ---------- |
| Warm-up | 10s | 10 |
| Subida leve | 10s | 50 |
| Carga alta | 20s | 350 |
| Carga máxima | 20s | 650 |

---

## Sistema de Pontuação

| Resultado | Pontos |
| --------- | ------- |
| TP — fraude corretamente negada | +1 |
| TN — legítima corretamente aprovada | +1 |
| FP — legítima incorretamente negada | -1 |
| FN — fraude incorretamente aprovada | **-3** |
| Erro HTTP | **-5** |

```bash
final_score = max(0, accuracy) × (TARGET_P99_MS / max(p99, TARGET_P99_MS))
```

**TARGET_P99_MS = 10ms.** Acima disso o multiplicador de latência degrada linearmente.

---

## MCC Risk — Valores Conhecidos

| MCC | Categoria | Risco |
| ----- | ----------- | ------- |
| 7995 | Apostas / Cassino | 0.85 |
| 7801 | Loterias governamentais | 0.80 |
| 7802 | Corridas de cavalos | 0.75 |
| 5944 | Joalherias | 0.45 |
| 4511 | Companhias aéreas | 0.35 |
| 5812 | Restaurantes | 0.30 |
| 5311 | Lojas de departamento | 0.25 |
| 5912 | Farmácias | 0.20 |
| 5411 | Supermercados | 0.15 |
| 5999 | Varejo diverso | 0.50 |
| (desconhecido) | — | **0.50** |

---

## Constantes de Normalização

```json
{
  "max_amount": 10000,
  "max_installments": 12,
  "amount_vs_avg_ratio": 10,
  "max_minutes": 1440,
  "max_km": 1000,
  "max_tx_count_24h": 20,
  "max_merchant_avg_amount": 10000
}
```
