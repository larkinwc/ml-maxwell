#!/usr/bin/env python3
"""Idempotent patch: register-only software __dp4a for Maxwell (sm_50/52).

The previous emulation took the address of register ints (`(const int8_t*)&a`),
forcing the compiler to spill a/b to local memory and issue 4 byte-loads inside
the innermost MMVQ loop (62% of decode time). Replace with an address-free
arithmetic-shift version: each lane is sign-extended via `(x << k) >> 24`
(signed int shifts are arithmetic in CUDA C++), so everything stays in
registers and lowers to SHF/IMAD with no local-memory traffic.

Semantics identical to hardware __dp4a: signed 8-bit lanes, int accumulate.
"""
import sys

F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/csrc/libtorch_stable/quantization/gguf/vecdotq.cuh"
MARKER = "// DP4A-REGONLY"

src = open(F).read()
if MARKER in src:
    print("already patched")
    sys.exit(0)

old = """static __device__ __forceinline__ int ggml_dp4a_sw(const int a, const int b,
                                                    const int c) {
  const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
  return c + pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2] + pa[3] * pb[3];
}"""

new = """static __device__ __forceinline__ int ggml_dp4a_sw(const int a, const int b,
                                                    const int c) {
  // DP4A-REGONLY: address-free, register-only int8x4 dot product. Sign-extend
  // each lane with arithmetic shifts (signed int >> is arithmetic in CUDA),
  // avoiding the local-memory spill that `(const int8_t*)&a` forced on sm_50.
  int s = c;
  s += ((a << 24) >> 24) * ((b << 24) >> 24);  // lane 0
  s += ((a << 16) >> 24) * ((b << 16) >> 24);  // lane 1
  s += ((a <<  8) >> 24) * ((b <<  8) >> 24);  // lane 2
  s += ( a        >> 24) * ( b        >> 24);  // lane 3
  return s;
}"""

if src.count(old) != 1:
    print(f"ERROR: expected exactly 1 anchor, found {src.count(old)}", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
open(F, "w").write(src)
print("patched OK")
