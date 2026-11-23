// Copyright 2026 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "./crc32c_riscv.h"

// RISC-V Zvbc vector carry-less multiply implementation, compiled separately
// with -march=rv64gcv_zvl256b_zvbc_zbc so unsupported hosts never see these
// instructions.

#include <cstddef>
#include <cstdint>

#include "./crc32c_internal.h"
#include "crc32c/crc32c_config.h"

#if HAVE_RISCV_ZVBC

#include <riscv_vector.h>

namespace crc32c {
namespace {

// Folding constants for CRC32C (Castagnoli).
const uint64_t kFold[4] = {
    0x740eef02ULL, 0x9e4addf8ULL, 0xf20c0dfeULL, 0x493c7d27ULL};
constexpr uint64_t kC0 = 0xdd45aab8ULL;
constexpr uint64_t kC1 = 0x493c7d27ULL;
constexpr uint64_t kCQuo = 0xdea713f1ULL;
constexpr uint64_t kPoly = 0x105ec76f1ULL;
constexpr uint64_t kM32 = 0xffffffffULL;

// Barrett reduction of a 128-bit (lo, hi) state to 32-bit CRC.
inline uint32_t Barrett(uint64_t lo, uint64_t hi) {
  uint64_t r, rh, t0;
  asm volatile("clmul %0,%1,%2" : "=r"(r) : "r"(lo), "r"(kC1));
  asm volatile("clmulh %0,%1,%2" : "=r"(rh) : "r"(lo), "r"(kC1));
  uint64_t t1 = hi ^ r, t4 = t1 & kM32;
  t1 >>= 32;
  asm volatile("clmul %0,%1,%2" : "=r"(t0) : "r"(t4), "r"(kC0));
  uint64_t t3 = (rh << 32) ^ t1 ^ t0;
  t4 = t3 & kM32;
  asm volatile("clmul %0,%1,%2" : "=r"(t4) : "r"(t4), "r"(kCQuo));
  t4 &= kM32;
  asm volatile("clmul %0,%1,%2" : "=r"(t4) : "r"(t4), "r"(kPoly));
  t4 ^= t3;
  return static_cast<uint32_t>((t4 >> 32) & kM32);
}

}  // namespace

uint32_t ExtendRiscvZvbc(uint32_t crc, const uint8_t* data, size_t count) {
  const uint8_t* p = data;
  size_t n = count;

  // Short inputs: let the portable engine handle the API CRC convention.
  if (n < 64) return ExtendPortable(crc, data, count);

  uint32_t c = crc ^ kCRC32Xor;  // internal (pre/post-xored) state
  {

    // RVV strided loads (stride 16B) deinterleave each 64B block into the
    // even/odd 64-bit lane pairs without a memcpy round-trip.  RVV loads are
    // unaligned, so no pre-alignment byte loop is needed.
    vuint64m2_t vlo = __riscv_vlse64_v_u64m2(
        reinterpret_cast<const unsigned long*>(p), 16, 4);
    vuint64m2_t vhi = __riscv_vlse64_v_u64m2(
        reinterpret_cast<const unsigned long*>(p + 8), 16, 4);
    {
      uint64_t tmp[4] = {c, 0, 0, 0};
      vuint64m2_t cv = __riscv_vle64_v_u64m2(tmp, 4);
      vlo = __riscv_vxor_vv_u64m2(vlo, cv, 4);
    }
    p += 64;
    n -= 64;
    const uint64_t k1 = kFold[0], k2 = kFold[1];
    vuint64m2_t k0v = __riscv_vmv_v_x_u64m2(k1, 4);
    vuint64m2_t k1v = __riscv_vmv_v_x_u64m2(k2, 4);
    while (n >= 64) {
      vuint64m2_t dlo = __riscv_vlse64_v_u64m2(
          reinterpret_cast<const unsigned long*>(p), 16, 4);
      vuint64m2_t dhi = __riscv_vlse64_v_u64m2(
          reinterpret_cast<const unsigned long*>(p + 8), 16, 4);
      vuint64m2_t lo = __riscv_vclmul_vv_u64m2(vlo, k0v, 4);
      lo = __riscv_vxor_vv_u64m2(lo, __riscv_vclmul_vv_u64m2(vhi, k1v, 4), 4);
      vuint64m2_t hi = __riscv_vclmulh_vv_u64m2(vlo, k0v, 4);
      hi = __riscv_vxor_vv_u64m2(hi, __riscv_vclmulh_vv_u64m2(vhi, k1v, 4), 4);
      vlo = __riscv_vxor_vv_u64m2(lo, dlo, 4);
      vhi = __riscv_vxor_vv_u64m2(hi, dhi, 4);
      p += 64;
      n -= 64;
    }
    // Reduce 4 lanes -> 1.
    uint64_t lo[4], hi[4];
    __riscv_vse64(lo, vlo, 4);
    __riscv_vse64(hi, vhi, 4);
    uint64_t x0 = lo[0], x1 = hi[0];
    for (int i = 1; i < 4; ++i) {
      uint64_t l, r, l2, r2;
      asm volatile("clmul %0,%1,%2" : "=r"(l) : "r"(x0), "r"(kFold[2]));
      asm volatile("clmulh %0,%1,%2" : "=r"(r) : "r"(x0), "r"(kFold[2]));
      asm volatile("clmul %0,%1,%2" : "=r"(l2) : "r"(x1), "r"(kFold[3]));
      asm volatile("clmulh %0,%1,%2" : "=r"(r2) : "r"(x1), "r"(kFold[3]));
      x0 = l ^ l2 ^ lo[i];
      x1 = r ^ r2 ^ hi[i];
    }
    c = Barrett(x0, x1);
  }

  // Tail: convert the internal Barrett state back to the API CRC convention
  // for the portable slicing-by-8 engine, then back to internal for return.
  if (n) c = ExtendPortable(c ^ kCRC32Xor, p, n) ^ kCRC32Xor;
  return c ^ kCRC32Xor;
}

}  // namespace crc32c

#endif  // HAVE_RISCV_ZVBC
