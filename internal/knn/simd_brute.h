#ifndef SIMD_BRUTE_H
#define SIMD_BRUTE_H

#include <stdint.h>

int brute_fraud_count_avx2(
    const int16_t *data_soa,  /* SoA layout: [14][N] int16 */
    const uint8_t *labels,    /* [N] uint8 (0=legit, 1=fraud) */
    int N,
    const int16_t query[14],
    int K,
    int *out_fraud_count
);

#endif
