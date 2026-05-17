# API

## Endpoints

O serviço expõe exatamente dois endpoints no runtime principal.

| Endpoint | Método | Resposta |
|---|---|---|
| `/ready` | `GET` | `200 OK` com corpo `OK` |
| `/fraud-score` | `POST` | `200 OK` com JSON `{approved, fraud_score}` |

## `GET /ready`

Usado por health check local e pelo `docker-compose`.

Resposta:

```text
OK
```

## `POST /fraud-score`

### Payload

```json
{
  "id": "tx-3576980410",
  "transaction": {
    "amount": 384.88,
    "installments": 3,
    "requested_at": "2026-03-11T20:23:35Z"
  },
  "customer": {
    "avg_amount": 769.76,
    "tx_count_24h": 3,
    "known_merchants": ["MERC-009", "MERC-001", "MERC-001"]
  },
  "merchant": {
    "id": "MERC-001",
    "mcc": "5912",
    "avg_amount": 298.95
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 13.7090520965
  },
  "last_transaction": {
    "timestamp": "2026-03-11T14:58:35Z",
    "km_from_current": 18.8626479774
  }
}
```

### Campos

| Campo | Tipo | Obrigatório |
|---|---|---|
| `id` | string | sim |
| `transaction.amount` | number | sim |
| `transaction.installments` | integer | sim |
| `transaction.requested_at` | string RFC3339 | sim |
| `customer.avg_amount` | number | sim |
| `customer.tx_count_24h` | integer | sim |
| `customer.known_merchants` | string[] | sim |
| `merchant.id` | string | sim |
| `merchant.mcc` | string | sim |
| `merchant.avg_amount` | number | sim |
| `terminal.is_online` | boolean | sim |
| `terminal.card_present` | boolean | sim |
| `terminal.km_from_home` | number | sim |
| `last_transaction` | object ou `null` | não |
| `last_transaction.timestamp` | string RFC3339 | quando presente |
| `last_transaction.km_from_current` | number | quando presente |

### Resposta

```json
{
  "approved": false,
  "fraud_score": 1.0
}
```

### Regras da resposta

- `fraud_score` assume apenas estes valores: `0.0`, `0.2`, `0.4`, `0.6`, `0.8`, `1.0`
- `approved=true` quando `fraudCount < 3`
- `approved=false` quando `fraudCount >= 3`

## Comportamento em Erro

O handler evita HTTP error no caminho de parse inválido.

Se `VectorizeJSON` falhar, a resposta atual é:

```json
{
  "approved": true,
  "fraud_score": 0.0
}
```

Com isso:

- o status HTTP continua `200 OK`
- o erro entra como erro de classificação potencial, não como erro HTTP

## Restrições Operacionais

- corpo máximo configurado no servidor: `4 KiB`
- timeout de leitura: `750ms`
- timeout de escrita: `750ms`
- `IdleTimeout`: `10s`

## Notas de Implementação

- O hot path atual usa `VectorizeJSON(ctx.PostBody())`, não `json.Unmarshal` da request inteira.
- As respostas são pré-computadas em memória em `internal/handler/fraud_score.go`.
- O serviço principal não usa prefilter GBDT no handler atual.
