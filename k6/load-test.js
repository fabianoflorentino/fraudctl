import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:9999';

const payloads = [
  {
    id: "tx-001",
    transaction: { amount: 500, installments: 3, requested_at: "2026-03-11T20:23:35Z" },
    customer: { avg_amount: 200, tx_count_24h: 5, known_merchants: ["m1", "m2"] },
    merchant: { id: "m1", mcc: "5411", avg_amount: 150 },
    terminal: { is_online: true, card_present: false, km_from_home: 5 },
    last_transaction: { timestamp: "2026-03-11T18:00:00Z", km_from_current: 10 }
  },
  {
    id: "tx-002",
    transaction: { amount: 5000, installments: 12, requested_at: "2026-03-11T02:30:00Z" },
    customer: { avg_amount: 200, tx_count_24h: 15, known_merchants: [] },
    merchant: { id: "m99", mcc: "7995", avg_amount: 8000 },
    terminal: { is_online: true, card_present: false, km_from_home: 200 },
    last_transaction: { timestamp: "2026-03-10T22:00:00Z", km_from_current: 150 }
  },
  {
    id: "tx-003",
    transaction: { amount: 50, installments: 1, requested_at: "2026-03-11T12:00:00Z" },
    customer: { avg_amount: 100, tx_count_24h: 2, known_merchants: ["m1"] },
    merchant: { id: "m1", mcc: "5411", avg_amount: 80 },
    terminal: { is_online: false, card_present: true, km_from_home: 1 },
  },
  {
    id: "tx-004",
    transaction: { amount: 2500, installments: 6, requested_at: "2026-03-11T23:45:00Z" },
    customer: { avg_amount: 300, tx_count_24h: 8, known_merchants: ["m1", "m2"] },
    merchant: { id: "m50", mcc: "5999", avg_amount: 2000 },
    terminal: { is_online: true, card_present: false, km_from_home: 50 },
    last_transaction: { timestamp: "2026-03-11T20:00:00Z", km_from_current: 30 }
  },
  {
    id: "tx-005",
    transaction: { amount: 150, installments: 1, requested_at: "2026-03-11T09:15:00Z" },
    customer: { avg_amount: 180, tx_count_24h: 1, known_merchants: ["m1", "m2", "m3"] },
    merchant: { id: "m2", mcc: "5812", avg_amount: 120 },
    terminal: { is_online: true, card_present: false, km_from_home: 2 },
    last_transaction: { timestamp: "2026-03-10T10:00:00Z", km_from_current: 5 }
  },
];

export const options = {
  stages: [
    { duration: '10s', target: 10 },
    { duration: '30s', target: 50 },
    { duration: '30s', target: 100 },
    { duration: '30s', target: 150 },
    { duration: '10s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(99)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const payload = JSON.stringify(payloads[Math.floor(Math.random() * payloads.length)]);

  const res = http.post(`${BASE_URL}/fraud-score`, payload, {
    headers: { 'Content-Type': 'application/json' },
  });

  check(res, {
    'status 200': (r) => r.status === 200,
    'valid json': (r) => r.json()?.fraud_score !== undefined,
    'fast response': (r) => r.timings.duration < 500,
  });
}
