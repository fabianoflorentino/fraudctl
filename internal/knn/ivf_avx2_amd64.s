#include "textflag.h"

// SoA block layout for DIM=14, blockSize=8:
//   block[d*8 + lane] as int16, so dimension d starts at byte offset d*16.
//
// STEP(qoff, boff):
//   qoff = d*2   (byte offset into query int16 array)
//   boff = d*16  (byte offset into SoA block — 8 × int16 = 16 bytes per dim)
//
// Computes: for each lane l in [0,8): accum[l] += (query[d] - block[d*8+l])^2
// Y0 holds int64 accumulators for lanes 0..3, Y1 for lanes 4..7.

#define STEP(qoff, boff) \
	VPBROADCASTW qoff(AX), X2; \
	VPMOVSXWD X2, Y2; \
	VMOVDQU boff(BX), X3; \
	VPMOVSXWD X3, Y3; \
	VPSUBD Y3, Y2, Y4; \
	VPMULLD Y4, Y4, Y4; \
	VPMOVSXDQ X4, Y5; \
	VEXTRACTI128 $1, Y4, X6; \
	VPMOVSXDQ X6, Y6; \
	VPADDQ Y5, Y0, Y0; \
	VPADDQ Y6, Y1, Y1

// func scanBlock8AVX2(query *int16, block unsafe.Pointer, out *uint64)
// Computes 8 squared-L2 distances and writes to out[0..7].
TEXT ·scanBlock8AVX2(SB), NOSPLIT, $0-24
	MOVQ query+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ out+16(FP), CX

	VPXOR Y0, Y0, Y0 // int64 accum lanes 0..3
	VPXOR Y1, Y1, Y1 // int64 accum lanes 4..7

	// All 14 dimensions; qoff = d*2, boff = d*16
	STEP( 0,   0) // dim 0
	STEP( 2,  16) // dim 1
	STEP( 4,  32) // dim 2
	STEP( 6,  48) // dim 3
	STEP( 8,  64) // dim 4
	STEP(10,  80) // dim 5
	STEP(12,  96) // dim 6
	STEP(14, 112) // dim 7
	STEP(16, 128) // dim 8
	STEP(18, 144) // dim 9
	STEP(20, 160) // dim 10
	STEP(22, 176) // dim 11
	STEP(24, 192) // dim 12
	STEP(26, 208) // dim 13

	VMOVDQU Y0,  0(CX)
	VMOVDQU Y1, 32(CX)
	VZEROUPPER
	RET

// BLOCK8(blockoff, outoff): process one SoA block at blockoff, write to outoff.
#define BLOCK8(blockoff, outoff) \
	VPXOR Y0, Y0, Y0; \
	VPXOR Y1, Y1, Y1; \
	STEP( 0, blockoff+  0); \
	STEP( 2, blockoff+ 16); \
	STEP( 4, blockoff+ 32); \
	STEP( 6, blockoff+ 48); \
	STEP( 8, blockoff+ 64); \
	STEP(10, blockoff+ 80); \
	STEP(12, blockoff+ 96); \
	STEP(14, blockoff+112); \
	STEP(16, blockoff+128); \
	STEP(18, blockoff+144); \
	STEP(20, blockoff+160); \
	STEP(22, blockoff+176); \
	STEP(24, blockoff+192); \
	STEP(26, blockoff+208); \
	VMOVDQU Y0, outoff+ 0(CX); \
	VMOVDQU Y1, outoff+32(CX)

// func scanBlock32AVX2(query *int16, block unsafe.Pointer, out *uint64)
// Processes 4 consecutive SoA blocks (32 vectors), writes out[0..31].
// Each block is blockStride=224 bytes apart; out slots are 8*8=64 bytes each.
TEXT ·scanBlock32AVX2(SB), NOSPLIT, $0-24
	MOVQ query+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ out+16(FP), CX

	BLOCK8(  0,   0)
	BLOCK8(224,  64)
	BLOCK8(448, 128)
	BLOCK8(672, 192)

	VZEROUPPER
	RET
