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

#ifndef __CUDACC_RTC__
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
#endif

namespace cgbn {

enum class SyncScope {
  kInstance = 1,
  kWarp = 2,
  kBlock = 3,
  kGrid = 4
};

struct cgbn_cuda_default_parameters_t {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;
};

/* forward declarations */
template<uint32_t tpi, class params>
struct CudaBnContext;

template<class context_t, uint32_t bits, SyncScope syncable=SyncScope::kInstance>
struct CudaBnEnv;

/* main classes */
template<uint32_t tpi, class params=cgbn_cuda_default_parameters_t>
struct CudaBnContext {
  static constexpr uint32_t TPB = params::TPB;
  static constexpr uint32_t TPI = tpi;
  static constexpr uint32_t MAX_ROTATION = params::MAX_ROTATION;
  static constexpr uint32_t SHM_LIMIT = params::SHM_LIMIT;
  static constexpr bool     CONSTANT_TIME = params::CONSTANT_TIME;
  static constexpr bool     is_gpu = true;

  const MonitorKind    _monitor;
  ErrorReport  *const  _report;
  const int32_t        _instance;
  uint32_t             *_scratch;

  __device__ __forceinline__ CudaBnContext();
  __device__ __forceinline__ CudaBnContext(MonitorKind type);
  __device__ __forceinline__ CudaBnContext(MonitorKind type, ErrorReport *report);
  __device__ __forceinline__ CudaBnContext(MonitorKind type, ErrorReport *report, uint32_t instance);

  __device__ __forceinline__ uint32_t *scratch() const;
  __device__ __forceinline__ bool      check_errors() const;
  __device__ __noinline__    void      report_error(Error error) const;

  template<class env_t>
  __device__ __forceinline__ env_t env() {
    env_t env(*this);
    return env; }
  template<uint32_t bits, SyncScope syncable>
  __device__ __forceinline__ CudaBnEnv<CudaBnContext, bits, syncable> env() {
    CudaBnEnv<CudaBnContext, bits, syncable> env(*this);
    return env; }
};

template<uint32_t tpi, typename params>
struct _ContextInfer<tpi, params, true> {
  using type = CudaBnContext<tpi, params>; };

template<class context_t, uint32_t bits, SyncScope syncable>
struct CudaBnEnv {

  // bits must be divisible by 32
  static constexpr uint32_t        BITS=bits;
  static constexpr uint32_t        TPB=context_t::TPB;
  static constexpr uint32_t        TPI=context_t::TPI;
  static constexpr uint32_t        MAX_ROTATION=context_t::MAX_ROTATION;
  static constexpr uint32_t        SHM_LIMIT=context_t::SHM_LIMIT;
  static constexpr bool            CONSTANT_TIME=context_t::CONSTANT_TIME;
  static constexpr SyncScope SYNCABLE=syncable;

  static constexpr uint32_t        LIMBS=(bits/32+TPI-1)/TPI;
  static constexpr uint32_t        LOCAL_LIMBS=((bits+32)/64+TPI-1)/TPI*TPI;
  static constexpr uint32_t        UNPADDED_BITS=TPI*LIMBS*32;
  static constexpr uint32_t        PADDING=bits/32%TPI;
  static constexpr uint32_t        PAD_THREAD=(BITS/32)/LIMBS;
  static constexpr uint32_t        PAD_LIMB=(BITS/32)%LIMBS;

  struct Reg {
    using parent_env_t = CudaBnEnv;
    uint32_t _limbs[LIMBS];
  };
  struct WideReg {
    Reg _low, _high;
  };
  struct LocalMem {
    public:
    uint64_t _limbs[LOCAL_LIMBS];
  };
  struct AccumReg {
    public:
    uint32_t _carry;
    uint32_t _limbs[LIMBS];
    __device__ __forceinline__ AccumReg();
  };

  const context_t &_context;

  __device__ __forceinline__ CudaBnEnv(const context_t &context);

  /* size conversion */
  template<typename src_ty>
  __device__ __forceinline__ void       set(Reg &r, const src_ty &source) const;

  /* bn arithmetic routines */
  __device__ __forceinline__ void       set(Reg &r, const Reg &a) const;
  __device__ __forceinline__ void       swap(Reg &r, Reg &a) const;
  __device__ __forceinline__ int32_t    add(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ int32_t    sub(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ int32_t    negate(Reg &r, const Reg &a) const;
  __device__ __forceinline__ void       mul(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ void       mul_high(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ void       sqr(Reg &r, const Reg &a) const;
  __device__ __forceinline__ void       sqr_high(Reg &r, const Reg &a) const;
  __device__ __forceinline__ void       div(Reg &q, const Reg &num, const Reg &denom) const;
  __device__ __forceinline__ void       rem(Reg &r, const Reg &num, const Reg &denom) const;
  __device__ __forceinline__ void       div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom) const;
  __device__ __forceinline__ void       sqrt(Reg &s, const Reg &a) const;
  __device__ __forceinline__ void       sqrt_rem(Reg &s, Reg &r, const Reg &a) const;
  __device__ __forceinline__ bool       equals(const Reg &a, const Reg &b) const;
  __device__ __forceinline__ int32_t    compare(const Reg &a, const Reg &b) const;
  __device__ __forceinline__ void       extract_bits(Reg &r, const Reg &a, const uint32_t start, const uint32_t len) const;
  __device__ __forceinline__ void       insert_bits(Reg &r, const Reg &a, const uint32_t start, const uint32_t len, const Reg &value) const;

  /* ui32 arithmetic routines*/
  __device__ __forceinline__ uint32_t   get_ui32(const Reg &a) const;
  __device__ __forceinline__ void       set_ui32(Reg &r, const uint32_t value) const;
  __device__ __forceinline__ int32_t    add_ui32(Reg &r, const Reg &a, const uint32_t add) const;
  __device__ __forceinline__ int32_t    sub_ui32(Reg &r, const Reg &a, const uint32_t sub) const;
  __device__ __forceinline__ uint32_t   mul_ui32(Reg &r, const Reg &a, const uint32_t mul) const;
  __device__ __forceinline__ uint32_t   div_ui32(Reg &r, const Reg &a, const uint32_t div) const;
  __device__ __forceinline__ uint32_t   rem_ui32(const Reg &a, const uint32_t div) const;
  __device__ __forceinline__ bool       equals_ui32(const Reg &a, const uint32_t value) const;
  __device__ __forceinline__ bool       all_equals_ui32(const Reg &a, const uint32_t value) const;
  __device__ __forceinline__ int32_t    compare_ui32(const Reg &a, const uint32_t value) const;
  __device__ __forceinline__ uint32_t   extract_bits_ui32(const Reg &a, const uint32_t start, const uint32_t len) const;
  __device__ __forceinline__ void       insert_bits_ui32(Reg &r, const Reg &a, const uint32_t start, const uint32_t len, const uint32_t value) const;
  __device__ __forceinline__ uint32_t   binary_inverse_ui32(const uint32_t n0) const;
  __device__ __forceinline__ uint32_t   gcd_ui32(const Reg &a, const uint32_t value) const;

  /* wide arithmetic routines */
  __device__ __forceinline__ void       mul_wide(WideReg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ void       sqr_wide(WideReg &r, const Reg &a) const;
  __device__ __forceinline__ void       div_wide(Reg &q, const WideReg &num, const Reg &denom) const;
  __device__ __forceinline__ void       rem_wide(Reg &r, const WideReg &num, const Reg &denom) const;
  __device__ __forceinline__ void       div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom) const;
  __device__ __forceinline__ void       sqrt_wide(Reg &s, const WideReg &a) const;
  __device__ __forceinline__ void       sqrt_rem_wide(Reg &s, WideReg &r, const WideReg &a) const;

  /* logical, shifting, masking */
  __device__ __forceinline__ void       bitwise_and(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ void       bitwise_ior(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ void       bitwise_xor(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ void       bitwise_complement(Reg &r, const Reg &a) const;
  __device__ __forceinline__ void       bitwise_select(Reg &r, const Reg &clear, const Reg &set, const Reg &select) const;
  __device__ __forceinline__ void       bitwise_mask_copy(Reg &r, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_and(Reg &r, const Reg &a, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_ior(Reg &r, const Reg &a, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_xor(Reg &r, const Reg &a, const int32_t numbits) const;
  __device__ __forceinline__ void       bitwise_mask_select(Reg &r, const Reg &clear, const Reg &set, int32_t numbits) const;
  __device__ __forceinline__ void       shift_left(Reg &r, const Reg &a, const uint32_t numbits) const;
  __device__ __forceinline__ void       shift_right(Reg &r, const Reg &a, const uint32_t numbits) const;
  __device__ __forceinline__ void       rotate_left(Reg &r, const Reg &a, const uint32_t numbits) const;
  __device__ __forceinline__ void       rotate_right(Reg &r, const Reg &a, const uint32_t numbits) const;

  /* bit counting */
  __device__ __forceinline__ uint32_t   pop_count(const Reg &a) const;
  __device__ __forceinline__ uint32_t   clz(const Reg &a) const;
  __device__ __forceinline__ uint32_t   ctz(const Reg &a) const;

  /* accumulator APIs */
  __device__ __forceinline__ int32_t    resolve(Reg &sum, const AccumReg &accumulator) const;
  __device__ __forceinline__ void       set(AccumReg &accumulator, const Reg &value) const;
  __device__ __forceinline__ void       add(AccumReg &accumulator, const Reg &value) const;
  __device__ __forceinline__ void       sub(AccumReg &accumulator, const Reg &value) const;
  __device__ __forceinline__ void       all_set_ui32(Reg &r, const uint32_t value) const;
  __device__ __forceinline__ void       set_ui32(AccumReg &accumulator, const uint32_t value) const;
  __device__ __forceinline__ void       add_ui32(AccumReg &accumulator, const uint32_t value) const;
  __device__ __forceinline__ void       sub_ui32(AccumReg &accumulator, const uint32_t value) const;

  /* math */
  __device__ __forceinline__ void       binary_inverse(Reg &r, const Reg &x) const;
  __device__ __forceinline__ void       gcd(Reg &r, const Reg &a, const Reg &b) const;
  __device__ __forceinline__ bool       modular_inverse(Reg &r, const Reg &x, const Reg &modulus) const;
  __device__ __forceinline__ void       modular_power(Reg &r, const Reg &x, const Reg &exponent, const Reg &modulus) const;

  /* fast division: common divisor / modulus */
  __device__ __forceinline__ uint32_t   bn2mont(Reg &mont, const Reg &bn, const Reg &n) const;
  __device__ __forceinline__ void       mont2bn(Reg &bn, const Reg &mont, const Reg &n, const uint32_t np0) const;
  __device__ __forceinline__ void       mont_mul(Reg &r, const Reg &a, const Reg &b, const Reg &n, const uint32_t np0) const;
  __device__ __forceinline__ void       mont_sqr(Reg &r, const Reg &a, const Reg &n, const uint32_t np0) const;
  __device__ __forceinline__ void       mont_reduce_wide(Reg &r, const WideReg &a, const Reg &n, const uint32_t np0) const;

  __device__ __forceinline__ uint32_t   barrett_approximation(Reg &approx, const Reg &denom) const;
  __device__ __forceinline__ void       barrett_div(Reg &q, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_rem(Reg &r, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_div_wide(Reg &q, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_rem_wide(Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __device__ __forceinline__ void       barrett_div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;

  /* load/store to global or shared memory */
  __device__ __forceinline__ void       load(Reg &r, Mem<bits> *const address) const;
  __device__ __forceinline__ void       store(Mem<bits> *address, const Reg &a) const;
  __device__ __forceinline__ void       load_shorter(Reg &dst, uint32_t *const src, uint32_t mem_limb_count) const;
  __device__ __forceinline__ void       store_shorter(uint32_t *dst, const Reg &src, uint32_t mem_limb_count) const;

  /* load/store to local memory */
  __device__ __forceinline__ void       load(Reg &r, LocalMem *const address) const;
  __device__ __forceinline__ void       store(LocalMem *address, const Reg &a) const;
};

} // namespace cgbn

#include "cgbn/impl_cuda.cuh"

/*
experimental:

  // faster shift and rotate by a constant number of bits
  template<uint32_t numbits> __device__ __forceinline__ void shift_left(Reg &r, const Reg &a);
  template<uint32_t numbits> __device__ __forceinline__ void shift_right(Reg &r, const Reg &a);
  template<uint32_t numbits> __device__ __forceinline__ void rotate_left(Reg &r, const Reg &a);
  template<uint32_t numbits> __device__ __forceinline__ void rotate_right(Reg &r, const Reg &a);

*/
