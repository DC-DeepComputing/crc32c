// Copyright 2026 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// RISC-V-specific code checking for Zvbc / Zbc availability.

#ifndef CRC32C_CRC32C_RISCV_CHECK_H_
#define CRC32C_CRC32C_RISCV_CHECK_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "crc32c/crc32c_config.h"

#if defined(__riscv) && defined(__riscv_xlen) && __riscv_xlen == 64
#include <unistd.h>
#endif

namespace crc32c {

inline bool CanUseRiscvZvbc() {
#if defined(__riscv) && defined(__riscv_xlen) && __riscv_xlen == 64
  /* Zvbc / Zbc are compile-time ISA extensions on RISC-V.  The kernel is
     only built when the compiler targets them; confirm the host ISA string
     advertises both, like the ARM implementation checks HWCAP. */
  FILE* f = fopen("/proc/cpuinfo", "r");
  if (!f) return false;
  char line[4096];
  bool ok = false;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "isa", 3)) continue;
    const char* isa = strchr(line, ':');
    if (!isa) break;
    ok = strstr(isa, "_zvbc") && strstr(isa, "_zbc");
    break;
  }
  fclose(f);
  return ok;
#else
  return false;
#endif
}

}  // namespace crc32c

#endif  // CRC32C_CRC32C_RISCV_CHECK_H_
