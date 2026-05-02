#ifndef SIMD_KNN_H
#define SIMD_KNN_H

#include <stdint.h>
#include <stddef.h>

/*
 * SoA-block IVF index for AVX2-optimized KNN search.
 *
 * Memory layout per cluster:
 *   - offsets[ci] .. offsets[ci+1] are block indices
 *   - Each block holds 8 vectors in SoA order:
 *     [d0_0..d0_7, d1_0..d1_7, ..., d13_0..d13_7] = 112 int16
 *   - Labels are stored per-block (8 bytes per block)
 *
 * Query: int16[14] (scale=10000, same as stored vectors)
 * Returns: number of fraud neighbors among top-K (K=5)
 */

int knn_fraud_count_avx2(
    const int16_t *blocks,
    const uint8_t *labels,
    const uint32_t *offsets,
    const float *centroids,
    int k,
    int nprobe,
    const int16_t query[14],
    int *out_fraud_count
);

#endif /* SIMD_KNN_H */
