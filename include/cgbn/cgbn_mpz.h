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

#include <stdio.h>
#include <stdlib.h>

#include <gmp.h>

#include "cgbn/cgbn_types.cuh"

namespace cgbn {

enum class ConvergenceKind {
  kInstance,
  kWarp,
  kBlock,
  kGrid,
};

// TODO use constexpr struct instead of typename to pass param?
struct cgbn_default_gmp_parameters_t {
  static constexpr uint32_t TPB=0;
};

/* forward declarations */
template<uint32_t tpi, class params>
struct GmpBnContext;

template<class context_t, uint32_t bits, ConvergenceKind convergence>
struct GmpBnEnv;

template<uint32_t tpi, class params=cgbn_default_gmp_parameters_t>
struct GmpBnContext {
  static constexpr uint32_t TPI= tpi;
  static constexpr bool     is_gpu= false;

  const MonitorKind   _monitor;
  ErrorReport   *const _report;
  int32_t                _instance;

  __host__ __device__ GmpBnContext();
  __host__ __device__ GmpBnContext(MonitorKind monitor);
  __host__ __device__ GmpBnContext(MonitorKind monitor, ErrorReport *report);
  __host__ __device__ GmpBnContext(MonitorKind monitor, ErrorReport *report, uint32_t instance);
  __host__ __device__ bool check_errors() const;
  __host__ __device__ void report_error(Error error) const;

  template<class env_t>
  env_t env() {
    env_t env(*this);
    return env; }
  template<uint32_t bits, ConvergenceKind convergence>
  GmpBnEnv<GmpBnContext, bits, convergence> env() {
    GmpBnEnv<GmpBnContext, bits, convergence> env(*this);
    return env;
  }
};

template<uint32_t tpi, typename params>
struct _ContextInfer<tpi, params, false> {
  using type = GmpBnContext<tpi, params>; };

template<class context_t, uint32_t bits, ConvergenceKind convergence=ConvergenceKind::kInstance>
struct GmpBnEnv {
  static constexpr uint32_t TPI=context_t::TPI;
  static constexpr uint32_t BITS=bits;
  static constexpr uint32_t LIMBS=(bits/32+TPI-1)/TPI;
  static constexpr uint32_t LOCAL_LIMBS=((bits+32)/64+TPI-1)/TPI*TPI;
  static constexpr uint32_t UNPADDED_BITS=TPI*LIMBS*32;

  struct Reg {
    typedef GmpBnEnv parent_env_t;
    mpz_t _z;

    __host__ __device__ Reg() {
// if we don't guard, we get lotsa "calling __host__ from __host__ __device__" function error
#ifndef __CUDA_ARCH__
      mpz_init(_z);
#endif
    }
    __host__ __device__ ~Reg() {
#ifndef __CUDA_ARCH__
      mpz_clear(_z);
#endif
    }
  };
  struct WideReg {
    Reg _low, _high;
  };
  struct LocalMem {
    mpz_t _z;
    __host__ __device__ LocalMem() {
#ifndef __CUDA_ARCH__
      mpz_init(_z);
#endif
    }
    __host__ __device__ ~LocalMem() {
#ifndef __CUDA_ARCH__
      mpz_clear(_z);
#endif
    }
  };
  struct AccumReg {
    mpz_t _z;
    __host__ __device__ AccumReg() {
#ifndef __CUDA_ARCH__
      mpz_init(_z);
#endif
    }
    __host__ __device__ ~AccumReg() {
#ifndef __CUDA_ARCH__
      mpz_clear(_z);
#endif
    }
  };

  const context_t &_context;

  __host__ __device__ GmpBnEnv(const context_t &context);

  /* size conversion */
  template<typename src_ty>
  __host__ __device__ void       set(Reg &r, const src_ty &source) const;

  /* set/get routines */
  __host__ __device__ void       set(Reg &r, const Reg &a) const;
  __host__ __device__ void       swap(Reg &r, Reg &a) const;
  __host__ __device__ void       extract_bits(Reg &r, const Reg &a, uint32_t start, uint32_t len) const;
  __host__ __device__ void       insert_bits(Reg &r, const Reg &a, uint32_t start, uint32_t len, const Reg &value) const;

  /* ui32 arithmetic routines*/
  __host__ __device__ uint32_t   get_ui32(const Reg &a) const;
  __host__ __device__ void       all_set_ui32(Reg &r, const uint32_t value) const;
  __host__ __device__ void       set_ui32(Reg &r, const uint32_t value) const;
  __host__ __device__ int32_t    add_ui32(Reg &r, const Reg &a, const uint32_t add) const;
  __host__ __device__ int32_t    sub_ui32(Reg &r, const Reg &a, const uint32_t sub) const;
  __host__ __device__ uint32_t   mul_ui32(Reg &r, const Reg &a, const uint32_t mul) const;
  __host__ __device__ uint32_t   div_ui32(Reg &r, const Reg &a, const uint32_t div) const;
  __host__ __device__ uint32_t   rem_ui32(const Reg &a, const uint32_t div) const;
  __host__ __device__ bool       all_equals_ui32(const Reg &a, const uint32_t value) const;
  __host__ __device__ bool       equals_ui32(const Reg &a, const uint32_t value) const;
  __host__ __device__ int32_t    compare_ui32(const Reg &a, const uint32_t value) const;
  __host__ __device__ uint32_t   extract_bits_ui32(const Reg &a, const uint32_t start, const uint32_t len) const;
  __host__ __device__ void       insert_bits_ui32(Reg &r, const Reg &a, const uint32_t start, const uint32_t len, const uint32_t value) const;
  __host__ __device__ uint32_t   binary_inverse_ui32(const uint32_t n0) const;
  __host__ __device__ uint32_t   gcd_ui32(const Reg &a, const uint32_t value) const;

  /* bn arithmetic routines */
  __host__ __device__ int32_t    add(Reg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ int32_t    sub(Reg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ int32_t    negate(Reg &r, const Reg &a) const;
  __host__ __device__ void       mul(Reg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ void       mul_high(Reg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ void       sqr(Reg &r, const Reg &a) const;
  __host__ __device__ void       sqr_high(Reg &r, const Reg &a) const;
  __host__ __device__ void       div(Reg &q, const Reg &num, const Reg &denom) const;
  __host__ __device__ void       rem(Reg &r, const Reg &num, const Reg &denom) const;
  __host__ __device__ void       div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom) const;
  __host__ __device__ void       sqrt(Reg &s, const Reg &a) const;
  __host__ __device__ void       sqrt_rem(Reg &s, Reg &r, const Reg &a) const;
  __host__ __device__ bool       equals(const Reg &a, const Reg &b) const;
  __host__ __device__ int32_t    compare(const Reg &a, const Reg &b) const;

  /* wide math routines */
  __host__ __device__ void       mul_wide(WideReg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ void       sqr_wide(WideReg &r, const Reg &a) const;
  __host__ __device__ void       div_wide(Reg &q, const WideReg &num, const Reg &denom) const;
  __host__ __device__ void       rem_wide(Reg &r, const WideReg &num, const Reg &denom) const;
  __host__ __device__ void       div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom) const;
  __host__ __device__ void       sqrt_wide(Reg &s, const WideReg &a) const;
  __host__ __device__ void       sqrt_rem_wide(Reg &s, WideReg &r, const WideReg &a) const;

  /* logical, shifting, masking */
  __host__ __device__ void       bitwise_complement(Reg &r, const Reg &a) const;
  __host__ __device__ void       bitwise_and(Reg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ void       bitwise_ior(Reg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ void       bitwise_xor(Reg &r, const Reg &a, const Reg &b) const;
  __host__ __device__ void       bitwise_select(Reg &r, const Reg &clear, const Reg &set, const Reg &select) const;
  __host__ __device__ void       bitwise_mask_copy(Reg &r, const int32_t numbits) const;
  __host__ __device__ void       bitwise_mask_and(Reg &r, const Reg &a, const int32_t numbits) const;
  __host__ __device__ void       bitwise_mask_ior(Reg &r, const Reg &a, const int32_t numbits) const;
  __host__ __device__ void       bitwise_mask_xor(Reg &r, const Reg &a, const int32_t numbits) const;
  __host__ __device__ void       bitwise_mask_select(Reg &r, const Reg &clear, const Reg &set, const int32_t numbits) const;
  __host__ __device__ void       shift_left(Reg &r, const Reg &a, const uint32_t numbits) const;
  __host__ __device__ void       shift_right(Reg &r, const Reg &a, const uint32_t numbits) const;
  __host__ __device__ void       rotate_left(Reg &r, const Reg &a, const uint32_t numbits) const;
  __host__ __device__ void       rotate_right(Reg &r, const Reg &a, const uint32_t numbits) const;

  /* faster shift and rotate by a constant number of bits */
  template<uint32_t numbits> __host__ __device__ void shift_left(Reg &r, const Reg &a) const;
  template<uint32_t numbits> __host__ __device__ void shift_right(Reg &r, const Reg &a) const;
  template<uint32_t numbits> __host__ __device__ void rotate_left(Reg &r, const Reg &a) const;
  template<uint32_t numbits> __host__ __device__ void rotate_right(Reg &r, const Reg &a) const;

  /* bit counting */
  __host__ __device__ uint32_t   pop_count(const Reg &a) const;
  __host__ __device__ uint32_t   clz(const Reg &a) const;
  __host__ __device__ uint32_t   ctz(const Reg &a) const;

  /* accumulator APIs */
  __host__ __device__ int32_t    resolve(Reg &sum, const AccumReg &accumulator) const;
  __host__ __device__ void       set_ui32(AccumReg &accumulator, const uint32_t value) const;
  __host__ __device__ void       add_ui32(AccumReg &accumulator, const uint32_t value) const;
  __host__ __device__ void       sub_ui32(AccumReg &accumulator, const uint32_t value) const;
  __host__ __device__ void       set(AccumReg &accumulator, const Reg &value) const;
  __host__ __device__ void       add(AccumReg &accumulator, const Reg &value) const;
  __host__ __device__ void       sub(AccumReg &accumulator, const Reg &value) const;

  /* math */
  __host__ __device__ void       binary_inverse(Reg &r, const Reg &m) const;
  __host__ __device__ bool       modular_inverse(Reg &r, const Reg &x, const Reg &modulus) const;
  __host__ __device__ void       modular_power(Reg &r, const Reg &a, const Reg &k, const Reg &m) const;
  __host__ __device__ void       gcd(Reg &r, const Reg &a, const Reg &b) const;

  /* fast division: common divisor / modulus */
  __host__ __device__ uint32_t   bn2mont(Reg &mont, const Reg &bn, const Reg &n) const;
  __host__ __device__ void       mont2bn(Reg &bn, const Reg &mont, const Reg &n, const uint32_t np0) const;
  __host__ __device__ void       mont_mul(Reg &r, const Reg &a, const Reg &b, const Reg &n, const uint32_t np0) const;
  __host__ __device__ void       mont_sqr(Reg &r, const Reg &a, const Reg &n, const uint32_t np0) const;
  __host__ __device__ void       mont_reduce_wide(Reg &r, const WideReg &a, const Reg &n, const uint32_t np0) const;

  __host__ __device__ uint32_t   barrett_approximation(Reg &approx, const Reg &denom) const;
  __host__ __device__ void       barrett_div(Reg &q, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __host__ __device__ void       barrett_rem(Reg &r, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __host__ __device__ void       barrett_div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __host__ __device__ void       barrett_div_wide(Reg &q, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __host__ __device__ void       barrett_rem_wide(Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;
  __host__ __device__ void       barrett_div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const;

  /* load/store to global or shared memory */
  __host__ __device__ void       load(Reg &r, Mem<bits> *const address) const;
  __host__ __device__ void       store(Mem<bits> *address, const Reg &a) const;
  __host__ __device__ void       load_shorter(Reg &dst, uint32_t *const src, uint32_t mem_limb_count) const;
  __host__ __device__ void       store_shorter(uint32_t *dst, const Reg &src, uint32_t mem_limb_count) const;

  /* load/store to local memory */
  __host__ __device__ void       load(Reg &r, LocalMem *const address) const;
  __host__ __device__ void       store(LocalMem *address, const Reg &a) const;
};
} // namespace cgbn

#ifndef __CUDA_ARCH__
#include "cgbn/impl_mpz.cc"
#endif
