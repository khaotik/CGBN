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

#if(__CUDACC_VER_MAJOR__<9 || (__CUDACC_VER_MAJOR__==9 && __CUDACC_VER_MINOR__<2))
  #if __CUDA_ARCH__>=700
    #error CGBN requires CUDA version 9.2 or above on Volta
  #endif
#endif


#include "cgbn/arith.cuh"
#include "cgbn/core/unpadded.cuh"
#include "cgbn/core.cuh"
#include "cgbn/core/core_singleton.cuh"
namespace cgbn {

/****************************************************************************************************************
 * CudaBnContext implementation for CUDA
 ****************************************************************************************************************/
template<uint32_t tpi, class params> __device__ __forceinline__
CudaBnContext<tpi, params>::CudaBnContext() : _monitor(MonitorKind::kNone), _report(NULL), _instance(0xFFFFFFFF) { }
template<uint32_t tpi, class params> __device__ __forceinline__
CudaBnContext<tpi, params>::CudaBnContext(MonitorKind monitor) : _monitor(monitor), _report(NULL), _instance(0xFFFFFFFF) {
  if(monitor!=MonitorKind::kNone) {
    if(tpi!=32 && tpi!=16 && tpi!=8 && tpi!=4)
      report_error(Error::kUnsupportedTPI);
    if(params::TPB!=0 && params::TPB!=blockDim.x)
      report_error(Error::kTBPMismatch);
    if(params::CONSTANT_TIME)
      report_error(Error::kUnsupportedOperation);
  }
}
template<uint32_t tpi, class params> __device__ __forceinline__
CudaBnContext<tpi, params>::CudaBnContext(MonitorKind monitor, ErrorReport *report) : _monitor(monitor), _report(report), _instance(0xFFFFFFFF) {
  if(monitor!=MonitorKind::kNone) {
    if(tpi!=32 && tpi!=16 && tpi!=8 && tpi!=4)
      report_error(Error::kUnsupportedTPI);
    if(params::TPB!=0 && params::TPB!=blockDim.x)
      report_error(Error::kTBPMismatch);
    if(params::CONSTANT_TIME)
      report_error(Error::kUnsupportedOperation);
  }
}
template<uint32_t tpi, class params> __device__ __forceinline__
CudaBnContext<tpi, params>::CudaBnContext(MonitorKind monitor, ErrorReport *report, uint32_t instance) : _monitor(monitor), _report(report), _instance(instance) {
  if(monitor!=MonitorKind::kNone) {
    if(tpi!=32 && tpi!=16 && tpi!=8 && tpi!=4)
      report_error(Error::kUnsupportedTPI);
    if(params::TPB!=0 && params::TPB!=blockDim.x)
      report_error(Error::kTBPMismatch);
    if(params::CONSTANT_TIME)
      report_error(Error::kUnsupportedOperation);
  }
}
template<uint32_t tpi, class params> __device__ __forceinline__
bool
CudaBnContext<tpi, params>::check_errors() const {
  return _monitor!=MonitorKind::kNone;
}
template<uint32_t tpi, class params> __device__ __noinline__
void
CudaBnContext<tpi, params>::report_error(Error error) const {
  if((threadIdx.x & tpi-1)==0) {
    if(_report!=NULL) {
      if(atomicCAS((uint32_t *)&(_report->_error), (uint32_t)Error::kSuccess, (uint32_t)error)==(uint32_t)Error::kSuccess) {
        _report->_instance=_instance;
        _report->_threadIdx=threadIdx;
        _report->_blockIdx=blockIdx; } }
    if(_monitor==MonitorKind::kPrint) {
      switch(_report->_error) {
        case Error::kUnsupportedTPI:
          printf("cgbn error: unsupported threads per instance\n");
          break;
        case Error::kUnsupportedSize:
          printf("cgbn error: unsupported size\n");
          break;
        case Error::kUnsupportedLimbsPerThread:
          printf("cgbn error: unsupported limbs per thread\n");
          break;
        case Error::kUnsupportedOperation:
          printf("cgbn error: unsupported operation\n");
          break;
        case Error::kTBPMismatch:
          printf("cgbn error: TPB does not match blockDim.x\n");
          break;
        case Error::kTPIMismatch:
          printf("cgbn errpr: TPI does not match env_t::TPI\n");
          break;
        case Error::kDivisionByZero:
          printf("cgbn error: division by zero on instance\n");
          break;
        case Error::kDivsionOverflow:
          printf("cgbn error: division overflow on instance\n");
          break;
        case Error::kMontgomeryModulusError:
          printf("cgbn error: division invalid montgomery modulus\n");
          break;
        case Error::kModulusNotOdd:
          printf("cgbn error: invalid modulus (it must be odd)\n");
          break;
        case Error::kInversionDoesNotExist:
          printf("cgbn error: inverse does not exist\n");
          break;
        default:
          printf("cgbn error: unknown error reported by instance\n");
          break;
      }
    }
    else if(_monitor==MonitorKind::kHalt) {
      __trap();
    }
  }
}

/*
template<uint32_t threads_per_instance, uint32_t threads_per_block> template<uint32_t bits>
__device__ __forceinline__ CudaBnEnv<CudaBnContext, bits> CudaBnContext<threads_per_instance, threads_per_block>::env() {
  CudaBnEnv<CudaBnContext, bits> env(this);

  return env;
}

template<uint32_t threads_per_instance, uint32_t threads_per_block> template<typename env_t>
  __device__ __forceinline__ CudaBnEnv<CudaBnContext, env_t::_bits> CudaBnContext<threads_per_instance, threads_per_block>::env() {
    return env<env_t::_bits>();
}
*/

/****************************************************************************************************************
 * CudaBnEnv implementation for CUDA
 ****************************************************************************************************************/

/* constructor */
template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
CudaBnEnv<context_t, bits, syncable>::CudaBnEnv(const context_t &context) : _context(context) {
  if(_context.check_errors()) {
    if(bits==0 || (bits & 0x1F)!=0) 
      _context.report_error(Error::kUnsupportedSize);
  }
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::set(Reg &r, const Reg &a) const {
  cgbn::core::core_t<CudaBnEnv>::set(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::swap(Reg &r, Reg &a) const {
  cgbn::core::core_t<CudaBnEnv>::swap(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> template<typename src_ty> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::set(Reg &r, const src_ty &source) const {
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t source_thread=0, source_limb=0, value;

  // TPI and TPB must match.  TPB matches automatically
  if(_context.check_errors()) {
    if(TPI!=src_ty::parent_env_t::TPI) {
      _context.report_error(Error::kTPIMismatch);
      return;
    }
  }
  
  sync=cgbn::core::core_t<CudaBnEnv>::sync_mask();
  cgbn::core::mpzero<LIMBS>(r._limbs);
  #pragma nounroll
  for(int32_t index=0;index<BITS/32;index++) {
    #pragma unroll
    for(int32_t limb=0;limb<src_ty::parent_env_t::LIMBS;limb++)
      if(limb==source_limb)
        value=source._limbs[limb];
    value=__shfl_sync(sync, value, source_thread, TPI);
    #pragma unroll
    for(int32_t limb=0;limb<LIMBS;limb++)
      if(group_thread*LIMBS+limb==index)
        r._limbs[limb]=value;
    source_limb++;
    if(source_limb==src_ty::parent_env_t::LIMBS) {
      source_limb=0;
      if(++source_thread==TPI)
        break;
    }
  }
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::extract_bits(Reg &r, const Reg &a, const uint32_t start, const uint32_t len) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  
  uint32_t local_len=len;
  
  if(start>=BITS) {
    cgbn::core::mpzero<LIMBS>(r._limbs);
    return;
  }
  
  local_len=cgbn::core::umin(local_len, BITS-start);
  
  core::rotate_right(r._limbs, a._limbs, start);
  core::bitwise_mask_and(r._limbs, r._limbs, local_len);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void CudaBnEnv<context_t, bits, syncable>::insert_bits(Reg &r, const Reg &a, const uint32_t start, const uint32_t len, const Reg &value) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;

  uint32_t local_len=len;
  uint32_t mask[LIMBS], temp[LIMBS];
  
  if(start>=BITS) {
    cgbn::core::mpset<LIMBS>(r._limbs, a._limbs);
    return;
  }
  
  local_len=cgbn::core::umin(local_len, BITS-start);
  
  core::rotate_left(temp, value._limbs, start);
  core::bitwise_mask_copy(mask, start+local_len);
  core::bitwise_mask_xor(mask, mask, start);
  core::bitwise_select(r._limbs, a._limbs, temp, mask);
}

/* ui32 routines */
template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::get_ui32(const Reg &a) const {
  return cgbn::core::core_t<CudaBnEnv>::get_ui32(a._limbs); }

template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::set_ui32(Reg &r, const uint32_t value) const {
  cgbn::core::core_t<CudaBnEnv>::set_ui32(r._limbs, value); }

template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::add_ui32(Reg &r, const Reg &a, const uint32_t add) const {
  return cgbn::core::core_t<CudaBnEnv>::add_ui32(r._limbs, a._limbs, add); }

template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::sub_ui32(Reg &r, const Reg &a, const uint32_t sub) const {
  return cgbn::core::core_t<CudaBnEnv>::sub_ui32(r._limbs, a._limbs, sub); }

template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__ uint32_t CudaBnEnv<context_t, bits, syncable>::mul_ui32(Reg &r, const Reg &a, const uint32_t mul) const {
  return cgbn::core::core_t<CudaBnEnv>::mul_ui32(r._limbs, a._limbs, mul);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::div_ui32(Reg &r, const Reg &a, const uint32_t div) const {
  if(div==0) {
    if(_context.check_errors()) 
      _context.report_error(Error::kDivisionByZero);
    return 0;
  }
  return cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::div_ui32(r._limbs, a._limbs, div);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::rem_ui32(const Reg &a, const uint32_t div) const {
  if(div==0) {
    if(_context.check_errors()) 
      _context.report_error(Error::kDivisionByZero);
    return 0;
  }
  return cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::rem_ui32(a._limbs, div);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
bool
CudaBnEnv<context_t, bits, syncable>::equals_ui32(const Reg &a, const uint32_t value) const {
  return cgbn::core::core_t<CudaBnEnv>::equals_ui32(a._limbs, value);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
bool
CudaBnEnv<context_t, bits, syncable>::all_equals_ui32(const Reg &a, const uint32_t value) const {
  return cgbn::core::core_t<CudaBnEnv>::all_equals_ui32(a._limbs, value);
}
template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::compare_ui32(const Reg &a, const uint32_t value) const {
  return cgbn::core::core_t<CudaBnEnv>::compare_ui32(a._limbs, value);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t CudaBnEnv<context_t, bits, syncable>::extract_bits_ui32(const Reg &a, const uint32_t start, const uint32_t len) const {
  return cgbn::core::core_t<CudaBnEnv>::extract_bits_ui32(a._limbs, start, len);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void CudaBnEnv<context_t, bits, syncable>::insert_bits_ui32(Reg &r, const Reg &a, const uint32_t start, const uint32_t len, const uint32_t value) const {
  cgbn::core::core_t<CudaBnEnv>::insert_bits_ui32(r._limbs, a._limbs, start, len, value);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::binary_inverse_ui32(const uint32_t x) const {
  if(_context.check_errors()) {
    if((x & 0x01)==0) {
      _context.report_error(Error::kInversionDoesNotExist);
      return 0;
    }
  }
  return cgbn::core::ubinary_inverse(x);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::gcd_ui32(const Reg &a, const uint32_t value) const {
  if(value==0)
    return 0;
  return cgbn::core::ugcd(value, rem_ui32(a, value));
}
/* bn arithmetic routines */
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::add(Reg &r, const Reg &a, const Reg &b) const {
  return cgbn::core::core_t<CudaBnEnv>::add(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::sub(Reg &r, const Reg &a, const Reg &b) const {
  return cgbn::core::core_t<CudaBnEnv>::sub(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::negate(Reg &r, const Reg &a) const {
  return cgbn::core::core_t<CudaBnEnv>::negate(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::mul(Reg &r, const Reg &a, const Reg &b) const {
  uint32_t add[LIMBS];
  cgbn::core::mpzero<LIMBS>(add);
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mul(r._limbs, a._limbs, b._limbs, add);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::mul_high(Reg &r, const Reg &a, const Reg &b) const {
  uint32_t add[LIMBS];
  cgbn::core::mpzero<LIMBS>(add);
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mul_high(r._limbs, a._limbs, b._limbs, add);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sqr(Reg &r, const Reg &a) const {
  uint32_t add[LIMBS];
  
  cgbn::core::mpzero<LIMBS>(add);
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mul(r._limbs, a._limbs, a._limbs, add);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sqr_high(Reg &r, const Reg &a) const {
  uint32_t add[LIMBS];
  cgbn::core::mpzero<LIMBS>(add);
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mul_high(r._limbs, a._limbs, a._limbs, add);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::div(Reg &q, const Reg &num, const Reg &denom) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;
  
  if(_context.check_errors()) {
    if(equals_ui32(denom, 0)) {
      _context.report_error(Error::kDivisionByZero);
      return; } }
  
  // division of padded values is the same as division of unpadded valuess
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_low, num._limbs, shift);
  core::bitwise_mask_and(num_high, num_low, shift);
  numthreads=TPI-core::clzt(num_high);
  singleton::div_wide(q._limbs, num_low, num_high, denom_local, numthreads);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::rem(Reg &r, const Reg &num, const Reg &denom) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(equals_ui32(denom, 0)) {
      _context.report_error(Error::kDivisionByZero);
      return; } }
  // division of padded values is the same as division of unpadded valuess
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_low, num._limbs, shift);
  core::bitwise_mask_and(num_high, num_low, shift);
  core::bitwise_xor(num_low, num_low, num_high);
  numthreads=TPI-core::clzt(num_high);
  singleton::rem_wide(r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(equals_ui32(denom, 0)) {
      _context.report_error(Error::kDivisionByZero);
      return; } }
  // division of padded values is the same as division of unpadded valuess
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_low, num._limbs, shift);
  core::bitwise_mask_and(num_high, num_low, shift);
  core::bitwise_xor(num_low, num_low, num_high);
  numthreads=TPI-core::clzt(num_high);
  singleton::div_rem_wide(q._limbs, r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sqrt(Reg &s, const Reg &a) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t shift, numthreads;
  uint32_t shifted[LIMBS];
  
  shift=core::clz(a._limbs);
  if(shift==UNPADDED_BITS) {
    cgbn::core::mpzero<LIMBS>(s._limbs);
    return;
  }
  numthreads=(UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);
  core::rotate_left(shifted, a._limbs, shift & 0xFFFFFFFE);
  singleton::sqrt(s._limbs, shifted, numthreads);
  shift=(shift>>1) % (LIMBS*32);
  core::shift_right(s._limbs, s._limbs, shift);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sqrt_rem(Reg &s, Reg &r, const Reg &a) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t shift, numthreads;
  uint32_t remainder[LIMBS], temp[LIMBS];
  
  shift=core::clz(a._limbs);
  if(shift==UNPADDED_BITS) {
    cgbn::core::mpzero<LIMBS>(s._limbs);
    cgbn::core::mpzero<LIMBS>(r._limbs);
    return; }
  numthreads=(UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);
  core::rotate_left(temp, a._limbs, shift & 0xFFFFFFFE);
  singleton::sqrt_rem(s._limbs, remainder, temp, numthreads);
  shift=(shift>>1) % (LIMBS*32);
  singleton::sqrt_resolve_rem(r._limbs, s._limbs, 0, remainder, shift);
  core::shift_right(s._limbs, s._limbs, shift);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
bool
CudaBnEnv<context_t, bits, syncable>::equals(const Reg &a, const Reg &b) const {
  return cgbn::core::core_t<CudaBnEnv>::equals(a._limbs, b._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::compare(const Reg &a, const Reg &b) const {
  return cgbn::core::core_t<CudaBnEnv>::compare(a._limbs, b._limbs);
}
/* wide arithmetic routines */
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::mul_wide(WideReg &r, const Reg &a, const Reg &b) const {
  uint32_t add[LIMBS];
  
  cgbn::core::mpzero<LIMBS>(add);
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mul_wide(r._low._limbs, r._high._limbs, a._limbs, b._limbs, add);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sqr_wide(WideReg &r, const Reg &a) const {
  uint32_t add[LIMBS];
  
  cgbn::core::mpzero<LIMBS>(add);
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mul_wide(r._low._limbs, r._high._limbs, a._limbs, a._limbs, add);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::div_wide(Reg &q, const WideReg &num, const Reg &denom) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(core::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return; } }
  
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_high, num._high._limbs, shift-(UNPADDED_BITS-BITS));
  core::rotate_left(num_low, num._low._limbs, shift);
  core::bitwise_mask_select(num_high, num_high, num_low, shift-(UNPADDED_BITS-BITS));
  numthreads=TPI-core::clzt(num_high);
  singleton::div_wide(q._limbs, num_low, num_high, denom_local, numthreads);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::rem_wide(Reg &r, const WideReg &num, const Reg &denom) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(core::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return; } }
  
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_high, num._high._limbs, shift-(UNPADDED_BITS-BITS));
  core::rotate_left(num_low, num._low._limbs, shift);
  core::bitwise_mask_select(num_high, num_high, num_low, shift-(UNPADDED_BITS-BITS));
  core::bitwise_mask_and(num_low, num_low, shift-UNPADDED_BITS);
  numthreads=TPI-core::clzt(num_high);
  singleton::rem_wide(r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t num_low[LIMBS], num_high[LIMBS], denom_local[LIMBS];
  uint32_t shift, numthreads;

  if(_context.check_errors()) {
    if(core::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return; } }
  shift=core::clz(denom._limbs);
  core::rotate_left(denom_local, denom._limbs, shift);
  core::rotate_left(num_high, num._high._limbs, shift-(UNPADDED_BITS-BITS));
  core::rotate_left(num_low, num._low._limbs, shift);
  core::bitwise_mask_select(num_high, num_high, num_low, shift-(UNPADDED_BITS-BITS));
  core::bitwise_mask_and(num_low, num_low, shift-UNPADDED_BITS);
  numthreads=TPI-core::clzt(num_high);
  singleton::div_rem_wide(q._limbs, r._limbs, num_low, num_high, denom_local, numthreads);
  core::rotate_right(r._limbs, r._limbs, shift);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sqrt_wide(Reg &s, const WideReg &a) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t clz_shift, shift, numthreads;
  uint32_t high_shifted[LIMBS], low_shifted[LIMBS];
  
  clz_shift=core::clz(a._high._limbs);
  if(clz_shift==UNPADDED_BITS) {
    clz_shift=core::clz(a._low._limbs);
    if(clz_shift==UNPADDED_BITS) {
      cgbn::core::mpzero<LIMBS>(s._limbs);
      return;
    }
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::core::mpset<LIMBS>(high_shifted, a._low._limbs);
    cgbn::core::mpzero<LIMBS>(low_shifted);
    shift=clz_shift + UNPADDED_BITS;
  }
  else {
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::core::mpset<LIMBS>(high_shifted, a._high._limbs);
    core::rotate_left(low_shifted, a._low._limbs, clz_shift+(UNPADDED_BITS-BITS));
    shift=clz_shift+UNPADDED_BITS-BITS;
  }
  numthreads=(2*UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);

  core::rotate_left(high_shifted, high_shifted, clz_shift);
  if(shift<2*UNPADDED_BITS-BITS) {
    core::bitwise_mask_select(high_shifted, high_shifted, low_shifted, clz_shift);
    core::bitwise_mask_and(low_shifted, low_shifted, (int32_t)(shift-UNPADDED_BITS));
  }

  singleton::sqrt_wide(s._limbs, low_shifted, high_shifted, numthreads);

  shift=(shift>>1) % (LIMBS*32);
  core::shift_right(s._limbs, s._limbs, shift);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sqrt_rem_wide(Reg &s, WideReg &r, const WideReg &a) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core_unpadded;
  typedef cgbn::core::core_t<CudaBnEnv> core_padded;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;

  uint32_t group_thread=threadIdx.x & TPI-1;
  uint32_t clz_shift, shift, numthreads, c;
  uint32_t remainder[LIMBS], high_shifted[LIMBS], low_shifted[LIMBS];

  clz_shift=core_unpadded::clz(a._high._limbs);
  if(clz_shift==UNPADDED_BITS) {
    clz_shift=core_unpadded::clz(a._low._limbs);
    if(clz_shift==UNPADDED_BITS) {
      cgbn::core::mpzero<LIMBS>(s._limbs);
      cgbn::core::mpzero<LIMBS>(r._low._limbs);
      cgbn::core::mpzero<LIMBS>(r._high._limbs);
      return;
    }
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::core::mpset<LIMBS>(high_shifted, a._low._limbs);
    cgbn::core::mpzero<LIMBS>(low_shifted);
    shift=clz_shift + UNPADDED_BITS;
  }
  else {
    clz_shift=clz_shift & 0xFFFFFFFE;
    cgbn::core::mpset<LIMBS>(high_shifted, a._high._limbs);
    core_unpadded::rotate_left(low_shifted, a._low._limbs, clz_shift+(UNPADDED_BITS-BITS));
    shift=clz_shift+UNPADDED_BITS-BITS;
  }
  numthreads=(2*UNPADDED_BITS+LIMBS*64-1-shift) / (LIMBS*64);

  core_unpadded::rotate_left(high_shifted, high_shifted, clz_shift);
  if(shift<2*UNPADDED_BITS-BITS) {
    core_unpadded::bitwise_mask_select(high_shifted, high_shifted, low_shifted, clz_shift);
    core_unpadded::bitwise_mask_and(low_shifted, low_shifted, (int32_t)(shift-UNPADDED_BITS));
  }

  c=singleton::sqrt_rem_wide(s._limbs, remainder, low_shifted, high_shifted, numthreads);

  shift=(shift>>1) % (LIMBS*32);
  if(shift==0) {
    if(UNPADDED_BITS!=BITS)
      c=core_padded::clear_carry(remainder);
    cgbn::core::mpset<LIMBS>(r._low._limbs, remainder);
    cgbn::core::mpzero<LIMBS>(r._high._limbs);
    r._high._limbs[0]=(group_thread==0) ? c : 0;
  }
  else {
    singleton::sqrt_resolve_rem(r._low._limbs, s._limbs, c, remainder, shift);
    cgbn::core::mpzero<LIMBS>(r._high._limbs);
    if(UNPADDED_BITS!=BITS) {
      c=core_padded::clear_carry(r._low._limbs);
      r._high._limbs[0]=(group_thread==0) ? c : 0;
    }
    core_unpadded::shift_right(s._limbs, s._limbs, shift);
  }
}

/* bit counting */
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::pop_count(const Reg &a) const {
  return cgbn::core::core_t<CudaBnEnv>::pop_count(a._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::clz(const Reg &a) const {
  return cgbn::core::core_t<CudaBnEnv>::clz(a._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::ctz(const Reg &a) const {
  return cgbn::core::core_t<CudaBnEnv>::ctz(a._limbs);
}


/* logical, shifting, masking */
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_complement(Reg &r, const Reg &a) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_complement(r._limbs, a._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_and(Reg &r, const Reg &a, const Reg &b) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_and(r._limbs, a._limbs, b._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_ior(Reg &r, const Reg &a, const Reg &b) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_ior(r._limbs, a._limbs, b._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_xor(Reg &r, const Reg &a, const Reg &b) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_xor(r._limbs, a._limbs, b._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_select(Reg &r, const Reg &clear, const Reg &set, const Reg &select) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_select(r._limbs, clear._limbs, set._limbs, select._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_mask_copy(Reg &r, const int32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_mask_copy(r._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_mask_and(Reg &r, const Reg &a, const int32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_mask_and(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_mask_ior(Reg &r, const Reg &a, const int32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_mask_ior(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_mask_xor(Reg &r, const Reg &a, const int32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_mask_xor(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::bitwise_mask_select(Reg &r, const Reg &clear, const Reg &set, const int32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::bitwise_mask_select(r._limbs, clear._limbs, set._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::shift_left(Reg &r, const Reg &a, const uint32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::shift_left(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::shift_right(Reg &r, const Reg &a, const uint32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::shift_right(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::rotate_left(Reg &r, const Reg &a, const uint32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::rotate_left(r._limbs, a._limbs, numbits);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::rotate_right(Reg &r, const Reg &a, const uint32_t numbits) const {
  cgbn::core::core_t<CudaBnEnv>::rotate_right(r._limbs, a._limbs, numbits);
}

#if 0
template<class context_t, uint32_t bits, SyncScope syncable> template<uint32_t numbits>
__device__ __forceinline__ void CudaBnEnv<context_t, bits, syncable>::shift_left(Reg &r, const Reg &a) const {
  fwshift_left_constant<LIMBS, numbits>(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> template<uint32_t numbits>
__device__ __forceinline__ void CudaBnEnv<context_t, bits, syncable>::shift_right(Reg &r, const Reg &a) const {
  fwshift_right_constant<LIMBS, numbits>(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> template<uint32_t numbits>
__device__ __forceinline__ void CudaBnEnv<context_t, bits, syncable>::rotate_left(Reg &r, const Reg &a) const {
  fwrotate_left_constant<LIMBS, numbits>(r._limbs, a._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> template<uint32_t numbits>
__device__ __forceinline__ void CudaBnEnv<context_t, bits, syncable>::rotate_right(Reg &r, const Reg &a) const {
  fwrotate_right_constant<LIMBS, numbits>(r._limbs, a._limbs);
}
#endif

/* accumulator APIs */

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
CudaBnEnv<context_t, bits, syncable>::AccumReg::AccumReg() {
  _carry=0;
  cgbn::core::mpzero<LIMBS>(_limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
int32_t
CudaBnEnv<context_t, bits, syncable>::resolve(Reg &sum, const AccumReg &accumulator) const {
  typedef cgbn::core::core_t<CudaBnEnv> core;

  uint32_t carry=accumulator._carry;
  int32_t  result;

  cgbn::core::mpset<LIMBS>(sum._limbs, accumulator._limbs);
  result=core::resolve_add(carry, sum._limbs);
  core::clear_padding(sum._limbs);
  return result;
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::all_set_ui32(Reg &r, const uint32_t value) const {
  if (PADDING == 0) {
    #pragma unroll
    for(int32_t index=0; index<LIMBS; index++) {
      r._limbs[index] = value;
    }
  } else {
    const int32_t group_thread=threadIdx.x & TPI-1;
    #pragma unroll
    for(int32_t index=0; index<LIMBS; index++) {
      if(group_thread*LIMBS+index < (BITS/32)) {
        r._limbs[index] = value;
      }
    }
  }
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::set_ui32(AccumReg &accumulator, const uint32_t value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  accumulator._carry=0;
  accumulator._limbs[0]=(group_thread==0) ? value : 0;
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    accumulator._limbs[index]=0;
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::add_ui32(AccumReg &accumulator, const uint32_t value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  cgbn::core::chain_t<> chain;
  accumulator._limbs[0]=chain.add(accumulator._limbs[0], (group_thread==0) ? value : 0);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    accumulator._limbs[index]=chain.add(accumulator._limbs[index], 0);
  accumulator._carry=chain.add(accumulator._carry, 0);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sub_ui32(AccumReg &accumulator, const uint32_t value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  cgbn::core::chain_t<> chain;
  chain.sub(0, group_thread);
  accumulator._limbs[0]=chain.sub(accumulator._limbs[0], (group_thread==0) ? value : 0);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    accumulator._limbs[index]=chain.sub(accumulator._limbs[index], 0);
    
  if(PADDING==0)
    accumulator._carry=chain.add(accumulator._carry, (group_thread==TPI-1) ? 0xFFFFFFFF : 0);
  else
    accumulator._carry=chain.add(accumulator._carry, 0);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::set(AccumReg &accumulator, const Reg &value) const {
  accumulator._carry=0;
  cgbn::core::mpset<LIMBS>(accumulator._limbs, value._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::add(AccumReg &accumulator, const Reg &value) const {
  cgbn::core::chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    accumulator._limbs[index]=chain.add(accumulator._limbs[index], value._limbs[index]);
  accumulator._carry=chain.add(accumulator._carry, 0);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::sub(AccumReg &accumulator, const Reg &value) const {
  uint32_t group_thread=threadIdx.x & TPI-1;

  cgbn::core::chain_t<> chain;
  chain.sub(0, group_thread);
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    accumulator._limbs[index]=chain.sub(accumulator._limbs[index], value._limbs[index]);

  if(PADDING==0)
    accumulator._carry=chain.add(accumulator._carry, (group_thread==TPI-1) ? 0xFFFFFFFF : 0);
  else
    accumulator._carry=chain.add(accumulator._carry, 0);
}

/* math */
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::binary_inverse(Reg &r, const Reg &x) const {
  uint32_t low;
  
  if(_context.check_errors()) {
    low=cgbn::core::core_t<CudaBnEnv>::get_ui32(x._limbs);
    if((low & 0x01)==0) {
      _context.report_error(Error::kInversionDoesNotExist);
      return;
    }
  }

  cgbn::core::core_t<CudaBnEnv>::binary_inverse(r._limbs, x._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
bool
CudaBnEnv<context_t, bits, syncable>::modular_inverse(Reg &r, const Reg &x, const Reg &m) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  
  return cgbn::core::core_t<unpadded>::modular_inverse(r._limbs, x._limbs, m._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::modular_power(Reg &r, const Reg &a, const Reg &k, const Reg &m) const {
  WideReg wide;
  Reg      current, square, approx;
  int32_t     bit, m_clz, last;

  // FIXME -- errors get checked again and again
  
  if(_context.check_errors()) {
    if(compare(a, m)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }

  set_ui32(current, 1);
  set(square, a);
  m_clz=barrett_approximation(approx, m);
  last=bits-1-clz(k);
  if(last==-1) {
    set_ui32(r, 1);
    return; }
  for(bit=0;bit<last;bit++) {
    if(extract_bits_ui32(k, bit, 1)==1) {
      mul_wide(wide, current, square);
      barrett_rem_wide(current, wide, m, approx, m_clz);
    }
    mul_wide(wide, square, square);
    barrett_rem_wide(square, wide, m, approx, m_clz);
  }
  mul_wide(wide, current, square);
  barrett_rem_wide(r, wide, m, approx, m_clz);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::gcd(Reg &r, const Reg &a, const Reg &b) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  
  cgbn::core::core_t<unpadded>::gcd(r._limbs, a._limbs, b._limbs);
}
/* fast division: common divisor / modulus */
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::bn2mont(Reg &mont, const Reg &bn, const Reg &n) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t num_low[LIMBS], num_high[LIMBS], n_local[LIMBS];
  uint32_t shift, low;

  low=core::get_ui32(n._limbs);
  
  if(_context.check_errors()) {
    if((low & 0x01)==0) {
      _context.report_error(Error::kModulusNotOdd);
      return 0;
    }
    if(compare(bn, n)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return 0;
    }
  }

  // for padded values, we use a larger R
  cgbn::core::mpzero<LIMBS>(num_low);
  shift=core::clz(n._limbs);
  core::rotate_left(n_local, n._limbs, shift);
  core::rotate_left(num_high, bn._limbs, shift);
  singleton::rem_wide(mont._limbs, num_low, num_high, n_local, TPI);
  core::shift_right(mont._limbs, mont._limbs, shift);
  return -cgbn::core::ubinary_inverse(low);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::mont2bn(Reg &bn, const Reg &mont, const Reg &n, const uint32_t np0) const {
  uint32_t zeros[LIMBS];

  cgbn::core::mpzero<LIMBS>(zeros);

  // mont_reduce_wide returns 0<=res<=n
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mont_reduce_wide(bn._limbs, mont._limbs, zeros, n._limbs, np0, true);

  // handle the case of res==n
  if(cgbn::core::core_t<CudaBnEnv>::equals(bn._limbs, n._limbs))  
    cgbn::core::mpzero<LIMBS>(bn._limbs);
}

template<class context_t, uint32_t bits, SyncScope syncable>
__device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::mont_mul(Reg &r, const Reg &a, const Reg &b, const Reg &n, const uint32_t np0) const {
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mont_mul(r._limbs, a._limbs, b._limbs, n._limbs, np0);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::mont_sqr(Reg &r, const Reg &a, const Reg &n, const uint32_t np0) const {
  cgbn::core::core_singleton_t<CudaBnEnv, LIMBS>::mont_mul(r._limbs, a._limbs, a._limbs, n._limbs, np0);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::mont_reduce_wide(Reg &r, const WideReg &a, const Reg &n, const uint32_t np0) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t low[LIMBS], high[LIMBS];
  
  cgbn::core::mpset<LIMBS>(low, a._low._limbs);
  cgbn::core::mpset<LIMBS>(high, a._high._limbs);
  
  if(PADDING!=0) {
    core::rotate_right(high, high, UNPADDED_BITS-BITS);
    core::bitwise_mask_select(low, high, low, BITS);
    core::bitwise_mask_and(high, high, BITS);
  }
  
  // mont_reduce_wide returns 0<=res<=n
  singleton::mont_reduce_wide(r._limbs, low, high, n._limbs, np0, false);

  // handle the case of res==n
  if(core::equals(r._limbs, n._limbs))  
    cgbn::core::mpzero<LIMBS>(r._limbs);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
uint32_t
CudaBnEnv<context_t, bits, syncable>::barrett_approximation(Reg &approx, const Reg &denom) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t shift, shifted[LIMBS], low[LIMBS], high[LIMBS];

  shift=core::clz(denom._limbs);
  if(_context.check_errors()) {
    if(shift==UNPADDED_BITS) {
      _context.report_error(Error::kDivisionByZero);
      return 0xFFFFFFFF;
    }
  }

  if(shift==UNPADDED_BITS)
    return 0xFFFFFFFF;

  core::rotate_left(shifted, denom._limbs, shift);
  
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    low[index]=0xFFFFFFFF;
    high[index]=~shifted[index];  // high=0xFFFFFFFF - shifted[index]
  }
  
  singleton::div_wide(approx._limbs, low, high, shifted, TPI);
  return shift;
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::barrett_div(Reg &q, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  sync=core::sync_mask();
  core::shift_right(high, num._limbs, UNPADDED_BITS-denom_clz);
  cgbn::core::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::core::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::core::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      quotient[index]=0xFFFFFFFF;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);
  
  word=-__shfl_sync(sync, high[0], 0, TPI);
  c=cgbn::core::mpsub<LIMBS>(low, num._limbs, low);
  word-=core::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::core::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core::fast_propagate_add(c, low);
  }
  core::sub_ui32(q._limbs, quotient, sub);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::barrett_rem(Reg &r, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c;

  sync=core::sync_mask();
  core::shift_right(high, num._limbs, UNPADDED_BITS-denom_clz);
  cgbn::core::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::core::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::core::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      quotient[index]=0xFFFFFFFF;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  word=-__shfl_sync(sync, high[0], 0, TPI);
  c=cgbn::core::mpsub<LIMBS>(low, num._limbs, low);
  word-=core::fast_propagate_sub(c, low);
  while(word!=0) {
    c=cgbn::core::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core::fast_propagate_add(c, low);
  }
  cgbn::core::mpset<LIMBS>(r._limbs, low);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::barrett_div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  sync=core::sync_mask();
  core::shift_right(high, num._limbs, UNPADDED_BITS-denom_clz);
  cgbn::core::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::core::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::core::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      quotient[index]=0xFFFFFFFF;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  word=-__shfl_sync(sync, high[0], 0, TPI);
  c=cgbn::core::mpsub<LIMBS>(low, num._limbs, low);
  word-=core::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::core::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core::fast_propagate_add(c, low);
  }
  core::sub_ui32(q._limbs, quotient, sub);
  cgbn::core::mpset<LIMBS>(r._limbs, low);
}

template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::barrett_div_wide(Reg &q, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core_unpadded;
  typedef cgbn::core::core_t<CudaBnEnv> core_padded;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
  
  uint32_t sync, group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  if(_context.check_errors()) {
    if(core_unpadded::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }

  sync=core_unpadded::sync_mask();
  word=__shfl_sync(sync, num._high._limbs[0], 0, TPI);
  core_unpadded::rotate_left(low, num._low._limbs, denom_clz);
  core_unpadded::rotate_left(high, num._high._limbs, denom_clz-(UNPADDED_BITS-BITS));
  core_unpadded::bitwise_mask_select(high, high, low, denom_clz-(UNPADDED_BITS-BITS));
  cgbn::core::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::core::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::core::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core_padded::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(PADDING==0)
        quotient[index]=0xFFFFFFFF;
      else
        quotient[index]=(group_base<BITS/32-index) ? 0xFFFFFFFF : 0;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  if(PADDING==0)
    word=word-__shfl_sync(sync, high[0], 0, TPI);
  else {
    word=word-__shfl_sync(sync, low[PAD_LIMB], PAD_THREAD, TPI);
    core_padded::clear_padding(low);
  }
    
  c=cgbn::core::mpsub<LIMBS>(low, num._low._limbs, low);
  word-=core_padded::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::core::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core_padded::fast_propagate_add(c, low);
  }
  core_unpadded::sub_ui32(q._limbs, quotient, sub);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::barrett_rem_wide(Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core_unpadded;
  typedef cgbn::core::core_t<CudaBnEnv> core_padded;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
    
  uint32_t sync, group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c;

  if(_context.check_errors()) {
    if(core_unpadded::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return; } }

  sync=core_unpadded::sync_mask();
  word=__shfl_sync(sync, num._high._limbs[0], 0, TPI);
  core_unpadded::rotate_left(low, num._low._limbs, denom_clz);
  core_unpadded::rotate_left(high, num._high._limbs, denom_clz-(UNPADDED_BITS-BITS));
  core_unpadded::bitwise_mask_select(high, high, low, denom_clz-(UNPADDED_BITS-BITS));
  cgbn::core::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::core::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::core::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core_padded::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(PADDING==0)
        quotient[index]=0xFFFFFFFF;
      else
        quotient[index]=(group_base<BITS/32-index) ? 0xFFFFFFFF : 0;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  if(PADDING==0)
    word=word-__shfl_sync(sync, high[0], 0, TPI);
  else {
    word=word-__shfl_sync(sync, low[PAD_LIMB], PAD_THREAD, TPI);
    core_padded::clear_padding(low);
  }
    
  c=cgbn::core::mpsub<LIMBS>(low, num._low._limbs, low);
  word-=core_padded::fast_propagate_sub(c, low);
  while(word!=0) {
    c=cgbn::core::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core_padded::fast_propagate_add(c, low);
  }
  cgbn::core::mpset<LIMBS>(r._limbs, low);
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::barrett_div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  typedef cgbn::core::unpadded_t<CudaBnEnv> unpadded;
  typedef cgbn::core::core_t<unpadded> core_unpadded;
  typedef cgbn::core::core_t<CudaBnEnv> core_padded;
  typedef cgbn::core::core_singleton_t<unpadded, LIMBS> singleton;
    
  uint32_t sync, group_thread=threadIdx.x & TPI-1, group_base=group_thread*LIMBS;
  uint32_t low[LIMBS], high[LIMBS], quotient[LIMBS], zero[LIMBS];
  uint32_t word, c, sub=0;

  if(_context.check_errors()) {
    if(core_unpadded::compare(num._high._limbs, denom._limbs)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }

  sync=core_unpadded::sync_mask();
  word=__shfl_sync(sync, num._high._limbs[0], 0, TPI);
  core_unpadded::rotate_left(low, num._low._limbs, denom_clz);
  core_unpadded::rotate_left(high, num._high._limbs, denom_clz-(UNPADDED_BITS-BITS));
  core_unpadded::bitwise_mask_select(high, high, low, denom_clz-(UNPADDED_BITS-BITS));
  cgbn::core::mpzero<LIMBS>(zero);
  singleton::mul_high(quotient, high, approx._limbs, zero);
  
  c=cgbn::core::mpadd<LIMBS>(quotient, quotient, high);
  c+=cgbn::core::mpadd32<LIMBS>(quotient, quotient, group_thread==0 ? 3 : 0);
  c=core_padded::resolve_add(c, quotient);
  
  if(c!=0) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      if(PADDING==0)
        quotient[index]=0xFFFFFFFF;
      else
        quotient[index]=(group_base<BITS/32-index) ? 0xFFFFFFFF : 0;
  }
  singleton::mul_wide(low, high, denom._limbs, quotient, zero);

  if(PADDING==0)
    word=word-__shfl_sync(sync, high[0], 0, TPI);
  else {
    word=word-__shfl_sync(sync, low[PAD_LIMB], PAD_THREAD, TPI);
    core_padded::clear_padding(low);
  }
    
  c=cgbn::core::mpsub<LIMBS>(low, num._low._limbs, low);
  word-=core_padded::fast_propagate_sub(c, low);
  while(word!=0) {
    sub++;
    c=cgbn::core::mpadd<LIMBS>(low, low, denom._limbs);
    word+=core_padded::fast_propagate_add(c, low);
  }
  core_unpadded::sub_ui32(q._limbs, quotient, sub);
  cgbn::core::mpset<LIMBS>(r._limbs, low);
}

/* load/store routines */
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::load(Reg &r, Mem<bits> *const address) const {
  int32_t group_thread=threadIdx.x & TPI-1;
  int32_t limb;

  #pragma unroll
  for(limb=0;limb<LIMBS;limb++) {
    if(PADDING!=0) {
      r._limbs[limb]=0;
      if(group_thread*LIMBS<BITS/32-limb) 
        r._limbs[limb]=address->_limbs[group_thread*LIMBS + limb];
    }
    else
      r._limbs[limb]=address->_limbs[group_thread*LIMBS + limb];
  }
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::store(Mem<bits> *address, const Reg &a) const {
  int32_t group_thread=threadIdx.x & TPI-1;
  int32_t limb;

  #pragma unroll
  for(limb=0;limb<LIMBS;limb++) {
    if(PADDING!=0) {
      if(group_thread*LIMBS<BITS/32-limb)
        address->_limbs[group_thread*LIMBS + limb]=a._limbs[limb];
#if 1
      else
        if(a._limbs[limb]!=0) {
          printf("BAD LIMB: %d %d %d\n", blockIdx.x, threadIdx.x, limb);
          __trap();
        }
#endif
    } else {
      address->_limbs[group_thread*LIMBS + limb]=a._limbs[limb];
    }
  }
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::load_shorter(Reg &dst, uint32_t *const src, uint32_t mem_limb_count) const {
  int32_t group_thread=threadIdx.x & TPI-1;
  int32_t limb;

  const uint32_t limb_todo = LIMBS < mem_limb_count ? LIMBS : mem_limb_count;
  #pragma unroll
  for(limb=0;limb<LIMBS;limb++) {
    bool pred = limb < limb_todo;
    if(PADDING!=0) {
      dst._limbs[limb]=0;
      if((group_thread*LIMBS<BITS/32-limb) && pred)
        dst._limbs[limb]=src[group_thread*LIMBS + limb];
    } else {
      dst._limbs[limb] = pred ? (src[group_thread*LIMBS + limb]) : 0;
    }
  }
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::store_shorter(uint32_t *dst, const Reg &src, uint32_t mem_limb_count) const {
  int32_t group_thread=threadIdx.x & TPI-1;
  int32_t limb;

  const uint32_t limb_todo = LIMBS < mem_limb_count ? LIMBS : mem_limb_count;
  #pragma unroll
  for(limb=0;limb<LIMBS;limb++) {
    uint32_t value = limb < limb_todo ? src._limbs[limb] : 0;
    if(PADDING!=0) {
      if(group_thread*LIMBS<BITS/32-limb) {
        dst[group_thread*LIMBS + limb] = value;
      }
#if 1
      else if(src._limbs[limb]!=0) {
        printf("BAD LIMB: %d %d %d\n", blockIdx.x, threadIdx.x, limb);
        __trap();
      }
#endif
    } else {
      dst[group_thread*LIMBS + limb] = value;
    }
  }
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::load(Reg &r, LocalMem *const address) const {
  int32_t limb;
  #pragma unroll
  for(limb=0;limb<LIMBS;limb++)
    r._limbs[limb]=address->_limbs[limb];
}
template<class context_t, uint32_t bits, SyncScope syncable> __device__ __forceinline__
void
CudaBnEnv<context_t, bits, syncable>::store(LocalMem *address, const Reg &a) const {
  int32_t limb;

  #pragma unroll
  for(limb=0;limb<LIMBS;limb++)
    address->_limbs[limb]=a._limbs[limb];
}

} // namespace cgbn
