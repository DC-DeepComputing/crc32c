// Copyright 2026 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// RISC-V Zvbc-specific code

#ifndef CRC32C_CRC32C_RISCV_H_
#define CRC32C_CRC32C_RISCV_H_

#include <cstddef>
#include <cstdint>

#include "crc32c/crc32c_config.h"

#if HAVE_RISCV_ZVBC

namespace crc32c {

uint32_t ExtendRiscvZvbc(uint32_t crc, const uint8_t* data, size_t count);

}  // namespace crc32c

#endif  // HAVE_RISCV_ZVBC

#endif  // CRC32C_CRC32C_RISCV_H_
