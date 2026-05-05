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
 * Returns: number of fraud neighbors among top-K (K=10)
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

/*
 * knn_fraud_count_retry: same as knn_fraud_count_avx2 but with boundary retry.
 *
 * If the fraud count after nprobe clusters is in [boundary_lo, boundary_hi],
 * scan retry_extra additional clusters (incrementally, keeping the same heap)
 * and return the updated fraud count.
 *
 * This is equivalent to thiagorigonatti's "Tier 1" boundary retry but done
 * entirely in C to avoid a second CGo round-trip.
 */
int knn_fraud_count_retry(
    const int16_t *blocks,
    const uint8_t *labels,
    const uint32_t *offsets,
    const float *centroids,
    int nlist,
    int nprobe,
    int retry_extra,
    int boundary_lo,
    int boundary_hi,
    const int16_t query[14],
    int *out_fraud_count
);

#endif /* SIMD_KNN_H */
