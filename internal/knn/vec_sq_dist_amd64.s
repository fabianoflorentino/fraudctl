#include "textflag.h"

// func vecSqDistAVX2(vec, q *int16) uint64
//
// Computes squared L2 distance between two [14]int16 vectors using AVX2.
// vec[0..13] and q[0..13] are sign-extended to int32, subtracted, squared,
// and accumulated. The result fits in uint32 but returned as uint64.
TEXT ·vecSqDistAVX2(SB), NOSPLIT, $0-24
	MOVQ vec+0(FP), AX
	MOVQ q+8(FP), BX

	// Load and process dims 0-7 (first 16 bytes)
	VPMOVSXWD (AX), Y0        // Y0 = int32(vec[0..7])
	VPMOVSXWD (BX), Y1        // Y1 = int32(q[0..7])
	VPSUBD Y1, Y0, Y0         // Y0 = diff[0..7]
	VPMULLD Y0, Y0, Y0        // Y0 = diff²[0..7]

	// Load and process dims 8-13 (bytes 16..27 = 12 bytes, load 16)
	VPMOVSXWD 16(AX), Y1      // Y1 = int32(vec[8..15]) — last 2 are garbage
	VPMOVSXWD 16(BX), Y2      // Y2 = int32(q[8..15])   — last 2 are garbage
	VPSUBD Y2, Y1, Y1         // Y1 = diff[8..15]
	VPMULLD Y1, Y1, Y1        // Y1 = diff²[8..15]

	// Zero out dims 14-15 (indices 6,7 in Y1) — they're garbage past the vector
	VPXOR Y2, Y2, Y2          // Y2 = zero
	VPBLENDD $0xC0, Y2, Y1, Y1 // keep Y1[0..5], replace Y1[6..7] with 0

	// Sum all partials
	VPADDD Y0, Y1, Y0         // Y0 = all diff²[0..15]

	// Horizontal sum of 8 int32 in Y0 → X0[0]
	VEXTRACTI128 $1, Y0, X1   // X1 = Y0[4..7]
	VPADDD X0, X1, X0         // X0 = [s0+s4, s1+s5, s2+s6, s3+s7]
	VPSHUFD $0xAE, X0, X1     // X1 = X0[2,3,0,1]
	VPADDD X0, X1, X0         // X0 = [s02+s13, ...]
	VPSHUFD $0x55, X0, X1     // X1 = X0[1,1,3,3]
	VPADDD X0, X1, X0         // X0[0] = grand total

	// Extract result
	VMOVD X0, AX              // AX = total (uint32)
	MOVQ AX, ret+16(FP)       // store as uint64

	VZEROUPPER
	RET

// func accumulateDotProductsAVX2(centroids []float32, nlist int, query [14]float32, out []float32)
//
// Computes out[ci] = sum over d of query[d] * centroids[d*nlist+ci] for ci in 0..nlist-1.
// Centroids are stored in SoA (Structure of Arrays) layout for cache-friendly access.
// Uses 8-wide AVX2 multiply-add.
TEXT ·accumulateDotProductsAVX2(SB), NOSPLIT, $0-112
	MOVQ centroids+0(FP), SI     // SI = centroids.ptr
	MOVQ nlist+24(FP), DX        // DX = nlist
	LEAQ query+32(FP), AX        // AX = &query[0]
	MOVQ out+88(FP), DI          // DI = out.ptr

	// Zero out[0..nlist-1] in 8-float blocks using 256-bit stores
	VPXOR Y0, Y0, Y0
	XORQ CX, CX
zero_loop:
	CMPQ CX, DX
	JAE zero_done
	VMOVUPS Y0, (DI)(CX*4)
	ADDQ $8, CX
	JMP zero_loop
zero_done:

	// Process each dimension
	XORQ R8, R8                // R8 = dim (0..13)
dim_loop:
	CMPQ R8, $14
	JAE done

	// Broadcast query[dim] to all 8 float32 lanes in Y1
	VBROADCASTSS (AX)(R8*4), Y1

	// Element offset = dim * nlist
	MOVQ R8, R9
	IMULQ DX, R9

	// Process centroids in blocks of 8
	XORQ CX, CX
inner_loop:
	CMPQ CX, DX
	JAE next_dim

	// Load 8 centroids and 8 out values
	VMOVUPS (SI)(R9*4), Y2      // Y2 = centroids[d*nlist+ci..ci+7]
	VMOVUPS (DI)(CX*4), Y3      // Y3 = out[ci..ci+7]

	// out[ci..ci+7] += centroids * query[dim]
	VMULPS Y1, Y2, Y2           // Y2 = centroids * qd
	VADDPS Y2, Y3, Y3           // Y3 = out + (centroids * qd)
	VMOVUPS Y3, (DI)(CX*4)      // store back

	ADDQ $8, CX
	ADDQ $8, R9
	JMP inner_loop

next_dim:
	INCQ R8
	JMP dim_loop

done:
	VZEROUPPER
	RET
