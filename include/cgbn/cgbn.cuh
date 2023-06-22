/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#pragma once
#define CGBN_API_INLINE __host__ __device__ __forceinline__


#ifndef __CUDACC_RTC__
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <type_traits>
#elif defined(CGBN_RTC_HAVE_STDLIB)
// pass
#else
using int8_t = char; static_assert(sizeof(int8_t) == 1);
using uint8_t = unsigned char; static_assert(sizeof(uint8_t) == 1);
using int16_t = short; static_assert(sizeof(int16_t) == 2);
using uint16_t = unsigned short; static_assert(sizeof(uint16_t) == 2);
using int32_t = int; static_assert(sizeof(int32_t) == 4);
using uint32_t = unsigned int; static_assert(sizeof(uint32_t) == 4);
using int64_t =  long long; static_assert(sizeof(int64_t) == 8);
using uint64_t = unsigned long long; static_assert(sizeof(uint64_t) == 8);
namespace std {
template<bool B, class T = void> struct enable_if {};
template<class T>                struct enable_if<true, T> { typedef T type; };
#if __cplusplus >= 201402L
template< bool B, class T = void > using enable_if_t = typename enable_if<B,T>::type;
#endif
} // namespace std
#endif // ifdef __CUDACC_RTC__

#include "cgbn/cgbn_types.cuh"

#ifndef __CUDACC_RTC__
#include "error_report.cuh"
#include "error_report.cu"
#endif

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
  #if !defined(XMP_IMAD) && !defined(XMP_XMAD) && !defined(XMP_WMAD)
     #if __CUDA_ARCH__<500
       #define XMP_IMAD
     #elif __CUDA_ARCH__<700
       #define XMP_XMAD
     #else
       #define XMP_WMAD
     #endif
  #endif
#endif
#include "cgbn/cgbn_cuda.cuh"

#if (defined(CGBN_NO_GMP) || defined(__CUDACC_RTC__))
template<typename ctx_ty, uint32_t bits, std::enable_if_t<ctx_ty::is_gpu,bool> = true >
using BnEnv = CudaBnEnv<ctx_ty, bits>;

#else

#include "cgbn/cgbn_mpz.h"
namespace cgbn {
template<typename ctx_ty, uint32_t bits> // TODO use variadic template for more args ?
using BnEnv = std::conditional_t<ctx_ty::is_gpu,
  CudaBnEnv<ctx_ty, bits>,
  GmpBnEnv<ctx_ty, bits>
>;
} // namespace cgbn
#endif

// TODO.feat impl CPU backend
#if 0
  #include "cgbn_cpu.h"
#endif

namespace cgbn {

template<class env_t, typename src_ty> CGBN_API_INLINE void
set(env_t env, typename env_t::Reg &r, const src_ty &a) {
  env.set(r, a); }
template<class env_t>
__host__ __device__ __forceinline__ void swap(env_t env, typename env_t::Reg &r, typename env_t::Reg &a) {
  env.swap(r, a); }
template<class env_t> CGBN_API_INLINE int32_t
add(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  return env.add(r, a, b); }
template<class env_t> CGBN_API_INLINE int32_t
sub(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  return env.sub(r, a, b); }
template<class env_t> CGBN_API_INLINE int32_t
negate(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a) {
  return env.negate(r, a); }
template<class env_t> CGBN_API_INLINE void
mul(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  env.mul(r, a, b); }
template<class env_t> CGBN_API_INLINE void
mul_high(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  env.mul_high(r, a, b); }
template<class env_t> CGBN_API_INLINE void
sqr(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a) {
  env.sqr(r, a); }
template<class env_t> CGBN_API_INLINE void
sqr_high(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a) {
  env.sqr_high(r, a); }
template<class env_t> CGBN_API_INLINE void
div(env_t env, typename env_t::Reg &q, const typename env_t::Reg &num, const typename env_t::Reg &denom) {
  env.div(q, num, denom); }
template<class env_t> CGBN_API_INLINE void
rem(env_t env, typename env_t::Reg &r, const typename env_t::Reg &num, const typename env_t::Reg &denom) {
  env.rem(r, num, denom); }
template<class env_t> CGBN_API_INLINE void
div_rem(env_t env, typename env_t::Reg &q, typename env_t::Reg &r, const typename env_t::Reg &num, const typename env_t::Reg &denom) {
  env.div_rem(q, r, num, denom); }
template<class env_t> CGBN_API_INLINE void
sqrt(env_t env, typename env_t::Reg &s, const typename env_t::Reg &a) {
  env.sqrt(s, a); }
template<class env_t> CGBN_API_INLINE void
sqrt_rem(env_t env, typename env_t::Reg &s, typename env_t::Reg &r, const typename env_t::Reg &a) {
  env.sqrt_rem(s, r, a); }
template<class env_t> CGBN_API_INLINE bool
equals(env_t env, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  return env.equals(a, b); }
template<class env_t> CGBN_API_INLINE int32_t
compare(env_t env, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  return env.compare(a, b); }
template<class env_t> CGBN_API_INLINE void
extract_bits(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t start, const uint32_t len) {
  env.extract_bits(r, a, start, len); }
template<class env_t> CGBN_API_INLINE void
insert_bits(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t start, const uint32_t len, const typename env_t::Reg &value) {
  env.insert_bits(r, a, start, len, value); }

/* ui32 arithmetic routines*/
template<class env_t> CGBN_API_INLINE uint32_t
get_ui32(env_t env, const typename env_t::Reg &a) {
  return env.get_ui32(a);
}
template<class env_t> CGBN_API_INLINE void
set_ui32(env_t env, typename env_t::Reg &r, const uint32_t value) {
  env.set_ui32(r, value);
}
template<class env_t> CGBN_API_INLINE int32_t
add_ui32(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t add) {
  return env.add_ui32(r, a, add);
}
template<class env_t> CGBN_API_INLINE int32_t
sub_ui32(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t sub) {
  return env.sub_ui32(r, a, sub);
}
template<class env_t> CGBN_API_INLINE uint32_t
mul_ui32(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t mul) {
  return env.mul_ui32(r, a, mul);
}
template<class env_t> CGBN_API_INLINE uint32_t
div_ui32(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t div) {
  return env.div_ui32(r, a, div);
}
template<class env_t> CGBN_API_INLINE uint32_t
rem_ui32(env_t env, const typename env_t::Reg &a, const uint32_t div) {
  return env.rem_ui32(a, div);
}
template<class env_t> CGBN_API_INLINE bool
equals_ui32(env_t env, const typename env_t::Reg &a, const uint32_t value) {
  return env.equals_ui32(a, value);
}
template<class env_t> CGBN_API_INLINE bool
all_equals_ui32(env_t env, const typename env_t::Reg &a, const uint32_t value) {
  return env.all_equals_ui32(a, value);
}
template<class env_t> CGBN_API_INLINE int32_t
compare_ui32(env_t env, const typename env_t::Reg &a, const uint32_t value) {
  return env.compare_ui32(a, value);
}
template<class env_t> CGBN_API_INLINE uint32_t
extract_bits_ui32(env_t env, const typename env_t::Reg &a, const uint32_t start, const uint32_t len) {
  return env.extract_bits_ui32(a, start, len);
}
template<class env_t> CGBN_API_INLINE void
insert_bits_ui32(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t start, const uint32_t len, const uint32_t value) {
  env.insert_bits_ui32(r, a, start, len, value);
}
template<class env_t> CGBN_API_INLINE uint32_t
binary_inverse_ui32(env_t env, const uint32_t n0) {
  return env.binary_inverse_ui32(n0);
}
template<class env_t> CGBN_API_INLINE uint32_t
gcd_ui32(env_t env, const typename env_t::Reg &a, const uint32_t value) {
  return env.gcd_ui32(a, value);
}
/* wide arithmetic routines */
template<class env_t> CGBN_API_INLINE void
mul_wide(env_t env, typename env_t::WideReg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  env.mul_wide(r, a, b);
}
template<class env_t> CGBN_API_INLINE void
sqr_wide(env_t env, typename env_t::WideReg &r, const typename env_t::Reg &a) {
  env.sqr_wide(r, a);
}
template<class env_t> CGBN_API_INLINE void
div_wide(env_t env, typename env_t::Reg &q, const typename env_t::WideReg &num, const typename env_t::Reg &denom) {
  env.div_wide(q, num, denom);
}
template<class env_t> CGBN_API_INLINE void
rem_wide(env_t env, typename env_t::Reg &r, const typename env_t::WideReg &num, const typename env_t::Reg &denom) {
  env.rem_wide(r, num, denom);
}
template<class env_t> CGBN_API_INLINE void
div_rem_wide(env_t env, typename env_t::Reg &q, typename env_t::Reg &r, const typename env_t::WideReg &num, const typename env_t::Reg &denom) {
  env.div_rem_wide(q, r, num, denom);
}
template<class env_t> CGBN_API_INLINE void
sqrt_wide(env_t env, typename env_t::Reg &s, const typename env_t::WideReg &a) {
  env.sqrt_wide(s, a);
}
template<class env_t> CGBN_API_INLINE void
sqrt_rem_wide(env_t env, typename env_t::Reg &s, typename env_t::WideReg &r, const typename env_t::WideReg &a) {
  env.sqrt_rem_wide(s, r, a);
}
/* logical, shifting, masking */
template<class env_t> CGBN_API_INLINE void
bitwise_and(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  env.bitwise_and(r, a, b);
}
template<class env_t> CGBN_API_INLINE void
bitwise_ior(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  env.bitwise_ior(r, a, b);
}
template<class env_t> CGBN_API_INLINE void
bitwise_xor(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  env.bitwise_xor(r, a, b);
}
template<class env_t> CGBN_API_INLINE void
bitwise_complement(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a) {
  env.bitwise_complement(r, a);
}
template<class env_t> CGBN_API_INLINE void
bitwise_select(env_t env, typename env_t::Reg &r, const typename env_t::Reg &clear, const typename env_t::Reg &set, const typename env_t::Reg &select) {
  env.bitwise_select(r, clear, set, select);
}
template<class env_t> CGBN_API_INLINE void
bitwise_mask_copy(env_t env, typename env_t::Reg &r, const int32_t numbits) {
  env.bitwise_mask_copy(r, numbits);
}
template<class env_t> CGBN_API_INLINE void
bitwise_mask_and(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const int32_t numbits) {
  env.bitwise_mask_and(r, a, numbits);
}
template<class env_t> CGBN_API_INLINE void
bitwise_mask_ior(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const int32_t numbits) {
  env.bitwise_mask_ior(r, a, numbits);
}
template<class env_t> CGBN_API_INLINE void
bitwise_mask_xor(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const int32_t numbits) {
  env.bitwise_mask_xor(r, a, numbits);
}
template<class env_t> CGBN_API_INLINE void
bitwise_mask_select(env_t env, typename env_t::Reg &r, const typename env_t::Reg &clear, const typename env_t::Reg &set, int32_t numbits) {
  env.bitwise_mask_select(r, clear, set, numbits);
}
template<class env_t> CGBN_API_INLINE void
shift_left(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t numbits) {
  env.shift_left(r, a, numbits);
}
template<class env_t> CGBN_API_INLINE void
shift_right(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t numbits) {
  env.shift_right(r, a, numbits);
}
template<class env_t> CGBN_API_INLINE void
rotate_left(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t numbits) {
  env.rotate_left(r, a, numbits);
}
template<class env_t> CGBN_API_INLINE void
rotate_right(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const uint32_t numbits) {
  env.rotate_right(r, a, numbits);
}
/* bit counting */
template<class env_t> CGBN_API_INLINE uint32_t
pop_count(env_t env, const typename env_t::Reg &a) {
  return env.pop_count(a);
}
template<class env_t> CGBN_API_INLINE uint32_t
clz(env_t env, const typename env_t::Reg &a) {
  return env.clz(a);
}
template<class env_t> CGBN_API_INLINE uint32_t
ctz(env_t env, const typename env_t::Reg &a) {
  return env.ctz(a);
}
/* accumulator APIs */
template<class env_t> CGBN_API_INLINE int32_t
resolve(env_t env, typename env_t::Reg &sum, const typename env_t::AccumReg &accumulator) {
  return env.resolve(sum, accumulator);
}
template<class env_t> CGBN_API_INLINE void
set(env_t env, typename env_t::AccumReg &accumulator, const typename env_t::Reg &value) {
  env.set(accumulator, value);
}
template<class env_t> CGBN_API_INLINE void
add(env_t env, typename env_t::AccumReg &accumulator, const typename env_t::Reg &value) {
  env.add(accumulator, value);
}
template<class env_t> CGBN_API_INLINE void
sub(env_t env, typename env_t::AccumReg &accumulator, const typename env_t::Reg &value) {
  env.sub(accumulator, value);
}
template<class env_t> CGBN_API_INLINE void
all_set_ui32(env_t env, typename env_t::Reg &r, const uint32_t value) {
  env.all_set_ui32(r, value);
}
template<class env_t> CGBN_API_INLINE void
set_ui32(env_t env, typename env_t::AccumReg &accumulator, const uint32_t value) {
  env.set_ui32(accumulator, value);
}
template<class env_t> CGBN_API_INLINE void
add_ui32(env_t env, typename env_t::AccumReg &accumulator, const uint32_t value) {
  env.add_ui32(accumulator, value);
}
template<class env_t> CGBN_API_INLINE void
sub_ui32(env_t env, typename env_t::AccumReg &accumulator, const uint32_t value) {
  env.sub_ui32(accumulator, value);
}
/* math */
template<class env_t> CGBN_API_INLINE void
binary_inverse(env_t env, typename env_t::Reg &r, const typename env_t::Reg &x) {
  env.binary_inverse(r, x);
}
template<class env_t> CGBN_API_INLINE void
gcd(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b) {
  env.gcd(r, a, b);
}
template<class env_t> CGBN_API_INLINE bool
modular_inverse(env_t env, typename env_t::Reg &r, const typename env_t::Reg &x, const typename env_t::Reg &modulus) {
  return env.modular_inverse(r, x, modulus);
}
template<class env_t> CGBN_API_INLINE void
modular_power(env_t env, typename env_t::Reg &r, const typename env_t::Reg &x, const typename env_t::Reg &exponent, const typename env_t::Reg &modulus) {
  env.modular_power(r, x, exponent, modulus);
}
/* fast division: common divisor / modulus */
template<class env_t> CGBN_API_INLINE uint32_t
bn2mont(env_t env, typename env_t::Reg &mont, const typename env_t::Reg &bn, const typename env_t::Reg &n) {
  return env.bn2mont(mont, bn, n);
}
template<class env_t> CGBN_API_INLINE void
mont2bn(env_t env, typename env_t::Reg &bn, const typename env_t::Reg &mont, const typename env_t::Reg &n, const uint32_t np0) {
  env.mont2bn(bn, mont, n, np0);
}
template<class env_t> CGBN_API_INLINE void
mont_mul(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &b, const typename env_t::Reg &n, const uint32_t np0) {
  env.mont_mul(r, a, b, n, np0);
}
template<class env_t> CGBN_API_INLINE void
mont_sqr(env_t env, typename env_t::Reg &r, const typename env_t::Reg &a, const typename env_t::Reg &n, const uint32_t np0) {
  env.mont_sqr(r, a, n, np0);
}
template<class env_t> CGBN_API_INLINE void
mont_reduce_wide(env_t env, typename env_t::Reg &r, const typename env_t::WideReg &a, const typename env_t::Reg &n, const uint32_t np0) {
  env.mont_reduce_wide(r, a, n, np0);
}
template<class env_t> CGBN_API_INLINE uint32_t
barrett_approximation(env_t env, typename env_t::Reg &approx, const typename env_t::Reg &denom) {
  return env.barrett_approximation(approx, denom);
}
template<class env_t> CGBN_API_INLINE void
barrett_div(env_t env, typename env_t::Reg &q, const typename env_t::Reg &num, const typename env_t::Reg &denom, const typename env_t::Reg &approx, const uint32_t denom_clz) {
  env.barrett_div(q, num, denom, approx, denom_clz);
}
template<class env_t> CGBN_API_INLINE void
barrett_rem(env_t env, typename env_t::Reg &r, const typename env_t::Reg &num, const typename env_t::Reg &denom, const typename env_t::Reg &approx, const uint32_t denom_clz) {
  env.barrett_rem(r, num, denom, approx, denom_clz);
}
template<class env_t> CGBN_API_INLINE void
barrett_div_rem(env_t env, typename env_t::Reg &q, typename env_t::Reg &r, const typename env_t::Reg &num, const typename env_t::Reg &denom, const typename env_t::Reg &approx, const uint32_t denom_clz) {
  env.barrett_div_rem(q, r, num, denom, approx, denom_clz);
}
template<class env_t> CGBN_API_INLINE void
barrett_div_wide(env_t env, typename env_t::Reg &q, const typename env_t::WideReg &num, const typename env_t::Reg &denom, const typename env_t::Reg &approx, const uint32_t denom_clz) {
  env.barrett_div_wide(q, num, denom, approx, denom_clz);
}
template<class env_t> CGBN_API_INLINE void
barrett_rem_wide(env_t env, typename env_t::Reg &r, const typename env_t::WideReg &num, const typename env_t::Reg &denom, const typename env_t::Reg &approx, const uint32_t denom_clz) {
  env.barrett_rem_wide(r, num, denom, approx, denom_clz);
}
template<class env_t> CGBN_API_INLINE void
barrett_div_rem_wide(env_t env, typename env_t::Reg &q, typename env_t::Reg &r, const typename env_t::WideReg &num, const typename env_t::Reg &denom, const typename env_t::Reg &approx, const uint32_t denom_clz) {
  env.barrett_div_rem_wide(q, r, num, denom, approx, denom_clz);
}
/* load/store to global or shared memory */
template<class env_t> CGBN_API_INLINE void
load(env_t env, typename env_t::Reg &r, Mem<env_t::BITS> *const address) {
  env.load(r, address);
}
template<class env_t> CGBN_API_INLINE void
store(env_t env, Mem<env_t::BITS> *address, const typename env_t::Reg &a) {
  env.store(address, a);
}
/* truncated load & store */
template<class env_t> CGBN_API_INLINE void
load_shorter(env_t env, typename env_t::Reg &dst, uint32_t *const src, uint32_t mem_limb_count) {
  env.load_shorter(dst, src, mem_limb_count);
}
template<class env_t> CGBN_API_INLINE void
store_shorter(env_t env, uint32_t *dst, const typename env_t::Reg &src, uint32_t mem_limb_count) {
  env.store_shorter(dst, src, mem_limb_count);
}
/* load/store to local memory */
template<class env_t> CGBN_API_INLINE void
load(env_t env, typename env_t::Reg &r, typename env_t::LocalMem *const address) {
  env.load(r, address);
}
template<class env_t> CGBN_API_INLINE void
store(env_t env, typename env_t::LocalMem *address, const typename env_t::Reg &a) {
  env.store(address, a);
}
} // namespace cgbn
