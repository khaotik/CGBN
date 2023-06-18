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

/****************************************************************************************************************
 * GmpBnContext implementation using GMP
 ****************************************************************************************************************/
namespace cgbn {
template<uint32_t tpi, class params>
GmpBnContext<tpi, params>::GmpBnContext():
  _monitor(MonitorKind::kNone),
  _report(nullptr), _instance(0xFFFFFFFF) {
}

template<uint32_t tpi, class params>
GmpBnContext<tpi, params>::GmpBnContext(MonitorKind monitor):
  _monitor(monitor),
  _report(nullptr),
  _instance(0xFFFFFFFF) {    
}

template<uint32_t tpi, class params>
GmpBnContext<tpi, params>::GmpBnContext(MonitorKind monitor, ErrorReport *report) : _monitor(monitor), _report(report), _instance(0xFFFFFFFF) {
}

template<uint32_t tpi, class params>
GmpBnContext<tpi, params>::GmpBnContext(MonitorKind monitor, ErrorReport *report, uint32_t instance) : _monitor(monitor), _report(report), _instance(instance) {
}

template<uint32_t tpi, class params>
bool GmpBnContext<tpi, params>::check_errors() const {
  return _monitor!=MonitorKind::kNone;
}

template<uint32_t tpi, class params>
void GmpBnContext<tpi, params>::report_error(Error error) const {
  if(_report!=nullptr) {
    dim3 maxdim;

    maxdim.x=0xFFFFFFFFu; maxdim.y=0xFFFFFFFFu; maxdim.z=0xFFFFFFFFu;
    if(__sync_bool_compare_and_swap(
          reinterpret_cast<volatile uint32_t*>(&(_report->_error)),
          static_cast<uint32_t>(Error::kSuccess),
          static_cast<uint32_t>(error))) {
      _report->_instance=_instance;
      _report->_threadIdx=maxdim;
      _report->_blockIdx=maxdim;
    }
  }
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
    exit(1);
  }
}

/*  forward declarations aren't working for this right now
template<uint32_t threads_per_instance, uint32_t threads_per_block> template<typename env_t>
env_t GmpBnContext<threads_per_instance, threads_per_block>::env() {
  env_t env(this);

  return env;
}

template<uint32_t threads_per_instance, uint32_t threads_per_block> template<uint32_t bits>
GmpBnEnv<GmpBnContext, bits> GmpBnContext<threads_per_instance, threads_per_block>::env() {
  GmpBnEnv<GmpBnContext, bits> env(this);

  return env;
}
*/

/****************************************************************************************************************
 * GmpBnEnv implementation using gmp
 ****************************************************************************************************************/

/* constructor */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
GmpBnEnv<context_t, bits, convergence>::GmpBnEnv(const context_t &context) : _context(context) {
}


/* size conversion */
template<class context_t, uint32_t bits, ConvergenceKind convergence>  template<typename src_ty>
void GmpBnEnv<context_t, bits, convergence>::set(Reg &r, const src_ty &source) const {
  mpz_fdiv_r_2exp(r._z, source._z, bits);
}


/* set get routines */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::set(Reg &r, const Reg &a) const {
  mpz_fdiv_r_2exp(r._z, a._z, bits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::swap(Reg &r, Reg &a) const {
  mpz_swap(r._z, a._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::extract_bits(Reg &r, const Reg &a, const uint32_t start, const uint32_t len) const {
  mpz_fdiv_q_2exp(r._z, a._z, start);
  mpz_fdiv_r_2exp(r._z, r._z, len);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::insert_bits(Reg &r, const Reg &a, const uint32_t start, const uint32_t len, const Reg &value) const {
  mpz_t   temp;
  int32_t index;

  if(start>=bits) {
    mpz_set(r._z, a._z);
    return;
  }

  mpz_init(temp);
  mpz_fdiv_r_2exp(temp, value._z, len);
  mpz_mul_2exp(temp, temp, start);
  mpz_set(r._z, a._z);
  for(index=start;index<start+len && index<bits;index++)
    mpz_clrbit(r._z, index);
  mpz_add(r._z, r._z, temp);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  mpz_clear(temp);
}


/* ui32 routines */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::get_ui32(const Reg &a) const {
  return mpz_get_ui(a._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::set_ui32(Reg &r, const uint32_t value) const {
   mpz_set_ui(r._z, value);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
int32_t GmpBnEnv<context_t, bits, convergence>::add_ui32(Reg &r, const Reg &a, const uint32_t add) const {
  uint32_t result;
  mpz_t    top;

  mpz_init(top);
  mpz_add_ui(r._z, a._z, add);
  mpz_fdiv_q_2exp(top, r._z, bits);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  result=mpz_get_ui(top);
  mpz_clear(top);
  return result;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
int32_t GmpBnEnv<context_t, bits, convergence>::sub_ui32(Reg &r, const Reg &a, const uint32_t sub) const {
  uint32_t result;
  mpz_t    top;

  mpz_init(top);
  mpz_sub_ui(r._z, a._z, sub);
  mpz_fdiv_q_2exp(top, r._z, bits);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  result=mpz_get_si(top);
  mpz_clear(top);
  return result;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::mul_ui32(Reg &r, const Reg &a, const uint32_t mul) const {
  uint32_t result;
  mpz_t    top;

  mpz_init(top);
  mpz_mul_ui(r._z, a._z, mul);
  mpz_fdiv_q_2exp(top, r._z, bits);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  result=mpz_get_ui(top);
  mpz_clear(top);
  return result;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::div_ui32(Reg &r, const Reg &a, const uint32_t div) const {
  uint32_t result;
  mpz_t    mod;

  if(_context.check_errors() && div==0) {
    _context.report_error(Error::kDivisionByZero);
    return 0;
  }

  mpz_init(mod);
  mpz_fdiv_r_ui(mod, a._z, div);
  mpz_fdiv_q_ui(r._z, a._z, div);
  result=mpz_get_ui(mod);
  mpz_clear(mod);
  return result;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::rem_ui32(const Reg &a, const uint32_t div) const {
  uint32_t result;
  mpz_t    mod;

  if(_context.check_errors() && div==0) {
    _context.report_error(Error::kDivisionByZero);
    return 0;
  }

  mpz_init(mod);
  mpz_fdiv_r_ui(mod, a._z, div);
  result=mpz_get_ui(mod);
  mpz_clear(mod);
  return result;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
bool GmpBnEnv<context_t, bits, convergence>::equals_ui32(const Reg &a, const uint32_t value) const {
  return mpz_cmp_ui(a._z, value)==0;
}

// TODO.feat
template<class context_t, uint32_t bits, ConvergenceKind convergence>
bool GmpBnEnv<context_t, bits, convergence>::all_equals_ui32(const Reg &a, const uint32_t value) const {
  bool ret=true;
  mpz_t b;
  mpz_init(b);
  uint32_t word;
  for(uint32_t i=0; i<bits; i+=32) {
    if (i==0) {
      word = mpz_get_ui(a._z);
    } else {
      mpz_fdiv_q_2exp(b, a._z, i);
      word = mpz_get_ui(b);
    }
    if (word != value) {
      ret = false;
      break;
    }
  }
  mpz_clear(b);
  return ret;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
int32_t GmpBnEnv<context_t, bits, convergence>::compare_ui32(const Reg &a, const uint32_t value) const {
  int32_t cmp=mpz_cmp_ui(a._z, value);

  if(cmp>0) return 1;
  if(cmp<0) return -1;
  return 0;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::extract_bits_ui32(const Reg &a, const uint32_t start, const uint32_t len) const {
  uint32_t result;
  mpz_t    chunk;

  mpz_init(chunk);
  mpz_fdiv_q_2exp(chunk, a._z, start);
  if(len<32)
    mpz_fdiv_r_2exp(chunk, chunk, len);
  result=mpz_get_ui(chunk);
  mpz_clear(chunk);
  return result;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::insert_bits_ui32(Reg &r, const Reg &a, const uint32_t start, const uint32_t len, const uint32_t value) const {
  int32_t index, local=len;
  mpz_t   chunk;

  if(start>bits) {
    mpz_set(r._z, a._z);
    return;
  }
  if(local>32)
    local=32;

  mpz_init(chunk);
  mpz_set_ui(chunk, value);
  mpz_fdiv_r_2exp(chunk, chunk, local);
  mpz_mul_2exp(chunk, chunk, start);

  mpz_set(r._z, a._z);
  for(index=start;index<start+local;index++)
    mpz_clrbit(r._z, index);
  mpz_add(r._z, r._z, chunk);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  mpz_clear(chunk);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::binary_inverse_ui32(const uint32_t n0) const {
  uint32_t inv=n0;

  inv=inv*(inv*n0+14);
  inv=inv*(inv*n0+2);
  inv=inv*(inv*n0+2);
  inv=inv*(inv*n0+2);
  return -inv;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::gcd_ui32(const Reg &a, const uint32_t value) const {
  uint32_t result;
  mpz_t    temp;

  if(value==0)
    return 0;

  mpz_init(temp);
  mpz_set_ui(temp, value);
  mpz_gcd(temp, temp, a._z);
  result=mpz_get_ui(temp);
  mpz_clear(temp);
  return result;
}


/* bn arithmetic routines */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
int32_t GmpBnEnv<context_t, bits, convergence>::add(Reg &r, const Reg &a, const Reg &b) const {
  int32_t carry;

  mpz_add(r._z, a._z, b._z);
  carry=mpz_tstbit(r._z, bits);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  return carry;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
int32_t GmpBnEnv<context_t, bits, convergence>::sub(Reg &r, const Reg &a, const Reg &b) const {
  int32_t carry;

  mpz_sub(r._z, a._z, b._z);
  carry=(mpz_sgn(r._z)==-1) ? -1 : 0;
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  return carry;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
int32_t GmpBnEnv<context_t, bits, convergence>::negate(Reg &r, const Reg &a) const {
  mpz_ui_sub(r._z, 0, a._z);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
  return (mpz_sgn(r._z)==0) ? 0 : -1;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::mul(Reg &r, const Reg &a, const Reg &b) const {
  mpz_mul(r._z, a._z, b._z);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::mul_high(Reg &r, const Reg &a, const Reg &b) const {
  mpz_mul(r._z, a._z, b._z);
  mpz_fdiv_q_2exp(r._z, r._z, bits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::sqr(Reg &r, const Reg &a) const {
  mpz_mul(r._z, a._z, a._z);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::sqr_high(Reg &r, const Reg &a) const {
  mpz_mul(r._z, a._z, a._z);
  mpz_fdiv_q_2exp(r._z, r._z, bits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::div(Reg &q, const Reg &num, const Reg &denom) const {
  mpz_fdiv_q(q._z, num._z, denom._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::rem(Reg &r, const Reg &num, const Reg &denom) const {
  mpz_fdiv_r(r._z, num._z, denom._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom) const {
  mpz_fdiv_qr(q._z, r._z, num._z, denom._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::sqrt(Reg &s, const Reg &a) const {
  mpz_sqrt(s._z, a._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::sqrt_rem(Reg &s, Reg &r, const Reg &a) const {
  mpz_t sqrt, rem;

  mpz_init(sqrt);
  mpz_init(rem);
  mpz_sqrt(sqrt, a._z);
  mpz_mul(rem, sqrt, sqrt);
  mpz_sub(r._z, a._z, rem);
  mpz_set(s._z, sqrt);
  mpz_clear(sqrt);
  mpz_clear(rem);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
bool GmpBnEnv<context_t, bits, convergence>::equals(const Reg &a, const Reg &b) const {
  return mpz_cmp(a._z, b._z)==0;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
int32_t GmpBnEnv<context_t, bits, convergence>::compare(const Reg &a, const Reg &b) const {
  int32_t cmp=mpz_cmp(a._z, b._z);

  if(cmp>0) return 1;
  if(cmp<0) return -1;
  return 0;
}


/* logical, shifting, masking */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_and(Reg &r, const Reg &a, const Reg &b) const {
  mpz_and(r._z, a._z, b._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_ior(Reg &r, const Reg &a, const Reg &b) const {
  mpz_ior(r._z, a._z, b._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_xor(Reg &r, const Reg &a, const Reg &b) const {
  mpz_xor(r._z, a._z, b._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_complement(Reg &r, const Reg &a) const {
  mpz_t temp;

  mpz_init(temp);
  mpz_set_ui(temp, 1);
  mpz_mul_2exp(temp, temp, bits);
  mpz_sub_ui(temp, temp, 1);
  mpz_xor(r._z, a._z, temp);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_select(Reg &r, const Reg &clear, const Reg &set, const Reg &select) const {
  mpz_t mask, temp;
  
  mpz_init(temp);
  mpz_init(mask);
  mpz_and(temp, set._z, select._z);
  mpz_set_ui(mask, 1);
  mpz_mul_2exp(mask, mask, bits);
  mpz_sub_ui(mask, mask, 1);
  mpz_xor(mask, mask, select._z);
  mpz_and(mask, mask, clear._z);
  mpz_ior(r._z, temp, mask);
  mpz_clear(mask);
  mpz_clear(temp);
}

template<uint32_t bits>
void make_mask(mpz_t mask, int32_t numbits) {
  if(numbits>=0 && numbits<bits) {
    mpz_set_ui(mask, 0);
    mpz_setbit(mask, numbits);
    mpz_sub_ui(mask, mask, 1);
  }
  else if(numbits<0 && numbits>-bits) {
    mpz_set_ui(mask, 0);
    mpz_setbit(mask, -numbits);
    mpz_sub_ui(mask, mask, 1);
    mpz_mul_2exp(mask, mask, bits+numbits);
  }
  else {
    mpz_set_si(mask, -1);
    mpz_fdiv_r_2exp(mask, mask, bits);
  }
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_mask_copy(Reg &r, const int32_t numbits) const {
  make_mask<bits>(r._z, numbits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_mask_and(Reg &r, const Reg &a, const int32_t numbits) const {
  mpz_t temp;

  mpz_init(temp);
  make_mask<bits>(temp, numbits);
  mpz_and(r._z, a._z, temp);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_mask_ior(Reg &r, const Reg &a, const int32_t numbits) const {
  mpz_t temp;

  mpz_init(temp);
  make_mask<bits>(temp, numbits);
  mpz_ior(r._z, a._z, temp);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_mask_xor(Reg &r, const Reg &a, const int32_t numbits) const {
  mpz_t temp;

  mpz_init(temp);
  make_mask<bits>(temp, numbits);
  mpz_xor(r._z, a._z, temp);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::bitwise_mask_select(Reg &r, const Reg &clear, const Reg &set, const int32_t numbits) const {
  mpz_t temp, mask, select;

  mpz_init(temp);
  mpz_init(mask);
  mpz_init(select);
  make_mask<bits>(select, numbits);

  mpz_and(temp, set._z, select);
  mpz_set_ui(mask, 1);
  mpz_mul_2exp(mask, mask, bits);
  mpz_sub_ui(mask, mask, 1);
  mpz_xor(select, select, mask);
  mpz_and(select, select, clear._z);
  mpz_ior(r._z, temp, select);
  
  mpz_clear(temp);
  mpz_clear(mask);
  mpz_clear(select);
}


template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::shift_left(Reg &r, const Reg &a, const uint32_t numbits) const {
  mpz_mul_2exp(r._z, a._z, numbits);
  mpz_fdiv_r_2exp(r._z, r._z, bits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::shift_right(Reg &r, const Reg &a, const uint32_t numbits) const {
  mpz_fdiv_q_2exp(r._z, a._z, numbits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::rotate_left(Reg &r, const Reg &a, const uint32_t numbits) const {
  mpz_t    left, right;
  uint32_t amount;

  amount=numbits % bits;
  mpz_init(left);
  mpz_init(right);
  mpz_mul_2exp(left, a._z, amount);
  mpz_fdiv_r_2exp(left, left, bits);
  mpz_fdiv_q_2exp(right, a._z, bits-amount);
  mpz_ior(r._z, left, right);
  mpz_clear(left);
  mpz_clear(right);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::rotate_right(Reg &r, const Reg &a, const uint32_t numbits) const {
  mpz_t    left, right;
  uint32_t amount;

  amount=bits-numbits % bits;
  if(amount==bits) {
    mpz_set(r._z, a._z);
    return;
  }
  mpz_init(left);
  mpz_init(right);
  mpz_mul_2exp(left, a._z, amount);
  mpz_fdiv_r_2exp(left, left, bits);
  mpz_fdiv_q_2exp(right, a._z, bits-amount);
  mpz_ior(r._z, left, right);
  mpz_clear(left);
  mpz_clear(right);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence> template<uint32_t numbits>
void GmpBnEnv<context_t, bits, convergence>::shift_left(Reg &r, const Reg &a) const {
  shift_left(r, a, numbits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence> template<uint32_t numbits>
void GmpBnEnv<context_t, bits, convergence>::shift_right(Reg &r, const Reg &a) const {
  shift_right(r, a, numbits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence> template<uint32_t numbits>
void GmpBnEnv<context_t, bits, convergence>::rotate_left(Reg &r, const Reg &a) const {
  rotate_left(r, a, numbits);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence> template<uint32_t numbits>
void GmpBnEnv<context_t, bits, convergence>::rotate_right(Reg &r, const Reg &a) const {
  rotate_right(r, a, numbits);
}

/* bit counting */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::pop_count(const Reg &a) const {
  return mpz_popcount(a._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::clz(const Reg &a) const {
  if(mpz_sgn(a._z)==0)
    return bits;
  return bits-mpz_sizeinbase(a._z, 2);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::ctz(const Reg &a) const {
  if(mpz_sgn(a._z)==0)
    return bits;
  return mpz_scan1(a._z, 0);
}



/* wide math routines */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::mul_wide(WideReg &r, const Reg &a, const Reg &b) const {
  mpz_t temp;

  mpz_init(temp);
  mpz_mul(temp, a._z, b._z);
  mpz_fdiv_r_2exp(r._low._z, temp, bits);
  mpz_fdiv_q_2exp(r._high._z, temp, bits);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::sqr_wide(WideReg &r, const Reg &a) const {
  mpz_t temp;

  mpz_init(temp);
  mpz_mul(temp, a._z, a._z);
  mpz_fdiv_r_2exp(r._low._z, temp, bits);
  mpz_fdiv_q_2exp(r._high._z, temp, bits);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::div_wide(Reg &q, const WideReg &num, const Reg &denom) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_sgn(denom._z)==0) {
      _context.report_error(Error::kDivisionByZero);
      return;
    }
    if(mpz_cmp(num._high._z, denom._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }
  mpz_init(temp);
  mpz_set(temp, num._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, num._low._z);
  mpz_fdiv_q(q._z, temp, denom._z);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::rem_wide(Reg &r, const WideReg &num, const Reg &denom) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_sgn(denom._z)==0) {
      _context.report_error(Error::kDivisionByZero);
      return;
    }
  }
  mpz_init(temp);
  mpz_set(temp, num._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, num._low._z);
  mpz_fdiv_r(r._z, temp, denom._z);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_sgn(denom._z)==0) {
      _context.report_error(Error::kDivisionByZero);
      return;
    }
    if(mpz_cmp(num._high._z, denom._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }
  mpz_init(temp);
  mpz_set(temp, num._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, num._low._z);
  mpz_fdiv_q(q._z, temp, denom._z);
  mpz_fdiv_r(r._z, temp, denom._z);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::sqrt_wide(Reg &s, const WideReg &a) const {
  mpz_t temp;

  mpz_init(temp);
  mpz_set(temp, a._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, a._low._z);
  mpz_sqrt(s._z, temp);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::sqrt_rem_wide(Reg &s, WideReg &r, const WideReg &a) const {
  mpz_t temp, sqrt;

  mpz_init(temp);
  mpz_init(sqrt);
  mpz_set(temp, a._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, a._low._z);
  mpz_sqrt(sqrt, temp);
  mpz_submul(temp, sqrt, sqrt);
  mpz_set(s._z, sqrt);
  mpz_fdiv_q_2exp(r._high._z, temp, bits);
  mpz_fdiv_r_2exp(r._low._z, temp, bits);
  mpz_clear(temp);
  mpz_clear(sqrt);
}

/* accumulator APIs */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ int32_t GmpBnEnv<context_t, bits, convergence>::resolve(Reg &sum, const AccumReg &accumulator) const {
  mpz_t    temp;
  uint32_t r;

  mpz_init(temp);
  mpz_fdiv_r_2exp(sum._z, accumulator._z, bits);
  mpz_fdiv_r_2exp(temp, accumulator._z, bits+32);
  mpz_fdiv_q_2exp(temp, temp, bits);
  r=mpz_get_ui(temp);
  mpz_clear(temp);
  return (int32_t)r;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ void GmpBnEnv<context_t, bits, convergence>::all_set_ui32(Reg &r, const uint32_t value) const {
  mpz_set_ui(r._z, value);
  for(int32_t i=32; i<bits; i+=32) {
    mpz_mul_2exp(r._z, r._z, 32);
    mpz_add_ui(r._z, r._z, value);
  }
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ void GmpBnEnv<context_t, bits, convergence>::set_ui32(AccumReg &accumulator, const uint32_t value) const {
  mpz_set_ui(accumulator._z, value);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ void GmpBnEnv<context_t, bits, convergence>::add_ui32(AccumReg &accumulator, const uint32_t value) const {
  mpz_add_ui(accumulator._z, accumulator._z, value);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ void GmpBnEnv<context_t, bits, convergence>::sub_ui32(AccumReg &accumulator, const uint32_t value) const {
  mpz_sub_ui(accumulator._z, accumulator._z, value);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ void GmpBnEnv<context_t, bits, convergence>::set(AccumReg &accumulator, const Reg &value) const {
  mpz_set(accumulator._z, value._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ void GmpBnEnv<context_t, bits, convergence>::add(AccumReg &accumulator, const Reg &value) const {
  mpz_add(accumulator._z, accumulator._z, value._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
__host__ void GmpBnEnv<context_t, bits, convergence>::sub(AccumReg &accumulator, const Reg &value) const {
  mpz_sub(accumulator._z, accumulator._z, value._z);
}


/* math */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::binary_inverse(Reg &r, const Reg &m) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_tstbit(m._z, 0)==0) {
      _context.report_error(Error::kModulusNotOdd);
      return;
    }
  }
  mpz_init(temp);
  mpz_setbit(temp, bits);
  mpz_invert(r._z, m._z, temp);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
bool GmpBnEnv<context_t, bits, convergence>::modular_inverse(Reg &r, const Reg &x, const Reg &modulus) const {
  uint32_t inverted;

  inverted=0;
  if(mpz_cmp_ui(modulus._z, 1)>0)
    inverted=mpz_invert(r._z, x._z, modulus._z);
  if(inverted==0)
    mpz_set_ui(r._z, 0);
  return inverted!=0;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::modular_power(Reg &r, const Reg &a, const Reg &k, const Reg &m) const {
  if(_context.check_errors()) {
    if(mpz_cmp(a._z, m._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }
  mpz_powm(r._z, a._z, k._z, m._z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::gcd(Reg &r, const Reg &a, const Reg &b) const {
  mpz_gcd(r._z, a._z, b._z);
}



/* fast division: common divisor / modulus */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::bn2mont(Reg &mont, const Reg &bn, const Reg &n) const {
  mpz_t    temp;
  uint32_t n0, inv;

  if(_context.check_errors()) {
    if(mpz_tstbit(n._z, 0)==0) {
      _context.report_error(Error::kModulusNotOdd);
      return 0;
    }
    if(mpz_cmp(bn._z, n._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return 0;
    }
  }
  mpz_init(temp);
  mpz_mul_2exp(temp, bn._z, UNPADDED_BITS);
  mpz_fdiv_r(mont._z, temp, n._z);
  mpz_clear(temp);

  n0=mpz_get_ui(n._z);
  inv=n0*(n0*n0+14);
  inv=inv*(inv*n0+2);
  inv=inv*(inv*n0+2);
  inv=inv*(inv*n0+2);
  return inv;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::mont2bn(Reg &bn, const Reg &mont, const Reg &n, uint32_t np0) const {
  mpz_t    prod, add;
  int32_t  index;
  uint32_t low;

  if(_context.check_errors()) {
    if(np0*(uint32_t)mpz_get_ui(n._z)!=0xFFFFFFFF) {
      _context.report_error(Error::kMontgomeryModulusError);
      return;
    }
  }
  mpz_init(prod);
  mpz_init(add);
  mpz_set(prod, mont._z);
  for(index=0;index<TPI*LIMBS;index++) {
    low=np0*(uint32_t)mpz_get_ui(prod);
    mpz_mul_ui(add, n._z, low);
    mpz_add(prod, prod, add);
    mpz_fdiv_q_2exp(prod, prod, 32);
  }
  if(mpz_cmp(prod, n._z)<0)
    mpz_set(bn._z, prod);
  else
    mpz_sub(bn._z, prod, n._z);
  mpz_clear(prod);
  mpz_clear(add);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::mont_mul(Reg &r, const Reg &a, const Reg &b, const Reg &n, uint32_t np0) const {
  mpz_t    prod, add;
  int32_t  index;
  uint32_t low;

  if(_context.check_errors()) {
    if(np0*(uint32_t)mpz_get_ui(n._z)!=0xFFFFFFFF) {
      _context.report_error(Error::kMontgomeryModulusError);
      return;
    }
  }
  mpz_init(prod);
  mpz_init(add);
  mpz_mul(prod, a._z, b._z);
  for(index=0;index<TPI*LIMBS;index++) {
    low=np0*(uint32_t)mpz_get_ui(prod);
    mpz_mul_ui(add, n._z, low);
    mpz_add(prod, prod, add);
    mpz_fdiv_q_2exp(prod, prod, 32);
  }
  if(mpz_tstbit(prod, bits)==1)
    mpz_sub(prod, prod, n._z);
  mpz_set(r._z, prod);
  mpz_clear(prod);
  mpz_clear(add);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::mont_sqr(Reg &r, const Reg &a, const Reg &n, uint32_t np0) const {
  mpz_t    prod, add;
  int32_t  index;
  uint32_t low;

  if(_context.check_errors()) {
    if(np0*(uint32_t)mpz_get_ui(n._z)!=0xFFFFFFFF) {
      _context.report_error(Error::kMontgomeryModulusError);
      return;
    }
  }
  mpz_init(prod);
  mpz_init(add);
  mpz_mul(prod, a._z, a._z);
  for(index=0;index<TPI*LIMBS;index++) {
    low=np0*(uint32_t)mpz_get_ui(prod);
    mpz_mul_ui(add, n._z, low);
    mpz_add(prod, prod, add);
    mpz_fdiv_q_2exp(prod, prod, 32);
  }
  if(mpz_tstbit(prod, bits)==1)
    mpz_sub(prod, prod, n._z);
  mpz_set(r._z, prod);
  mpz_clear(prod);
  mpz_clear(add);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::mont_reduce_wide(Reg &r, const WideReg &a, const Reg &n, uint32_t np0) const {
  mpz_t    prod, add;
  int32_t  index;
  uint32_t low;

  if(_context.check_errors()) {
    if(np0*(uint32_t)mpz_get_ui(n._z)!=0xFFFFFFFF) {
      _context.report_error(Error::kMontgomeryModulusError);
      return;
    }
  }
  mpz_init(prod);
  mpz_init(add);
  mpz_set(prod, a._high._z);
  mpz_mul_2exp(prod, prod, bits);
  mpz_add(prod, prod, a._low._z);
  for(index=0;index<TPI*LIMBS;index++) {
    low=np0*(uint32_t)mpz_get_ui(prod);
    mpz_mul_ui(add, n._z, low);
    mpz_add(prod, prod, add);
    mpz_fdiv_q_2exp(prod, prod, 32);
  }
  if(mpz_tstbit(prod, bits)==1)
    mpz_sub(prod, prod, n._z);
  if(mpz_cmp(prod, n._z)==0)
    mpz_set_ui(prod, 0);
  mpz_set(r._z, prod);
  mpz_clear(prod);
  mpz_clear(add);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
uint32_t GmpBnEnv<context_t, bits, convergence>::barrett_approximation(Reg &approx, const Reg &denom) const {
  mpz_t    temp, shifted;
  uint32_t clz;

  if(_context.check_errors()) {
    if(mpz_sgn(denom._z)==0) {
      _context.report_error(Error::kDivisionByZero);
      return 0xFFFFFFFF;
    }
  }

  if(mpz_sgn(denom._z)==0)
    return 0xFFFFFFFF;
  clz=bits-mpz_sizeinbase(denom._z, 2);

  mpz_init(temp);
  mpz_init(shifted);
  mpz_mul_2exp(shifted, denom._z, clz);
  mpz_setbit(temp, bits-1);
  if(mpz_cmp(temp, shifted)==0) {
    mpz_clrbit(temp, bits-1);
    mpz_setbit(temp, bits);
    mpz_sub_ui(approx._z, temp, 1);
  }
  else {
    mpz_set_ui(temp, 0);
    mpz_setbit(temp, 2*bits);
    mpz_fdiv_q(approx._z, temp, shifted);
    mpz_clrbit(approx._z, bits);
  }
  mpz_clear(temp);
  mpz_clear(shifted);
  return clz;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::barrett_div(Reg &q, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  mpz_t   temp, quot, rem;
  int32_t count;

  mpz_init(temp);
  mpz_init(quot);
  mpz_init(rem);

  // compute top
  mpz_fdiv_q_2exp(temp, num._z, bits-denom_clz);

  // q=(top * approx + top)>>bits
  mpz_mul(quot, temp, approx._z);
  mpz_fdiv_q_2exp(quot, quot, bits);
  mpz_add(quot, quot, temp);

  // q=min(q+3, beta-1), min might be unnecessary
  mpz_add_ui(quot, quot, 3);
  if(mpz_tstbit(quot, bits)==1) {
    mpz_set_ui(quot, 1);
    mpz_mul_2exp(quot, quot, bits);
    mpz_sub_ui(quot, quot, 1);
  }

  // r=num-q*denom
  mpz_mul(rem, quot, denom._z);
  mpz_sub(rem, num._z, rem);

  // correction steps
  count=0;
  while(count<3 && mpz_sgn(rem)==-1) {
    mpz_sub_ui(quot, quot, 1);
    mpz_add(rem, rem, denom._z);
    count++;
  }

  // check it
  mpz_fdiv_q(temp, num._z, denom._z);
  if(mpz_cmp(quot, temp)!=0) {
    printf("    num: ");
    mpz_out_str(stdout, 16, num._z);
    printf("\n");
    printf("  denom: ");
    mpz_out_str(stdout, 16, denom._z);
    printf("\n");
    printf(" approx: ");
    mpz_out_str(stdout, 16, approx._z);
    printf("\n");
    printf("    rem: ");
    mpz_out_str(stdout, 16, rem);
    printf("\n");
    printf("      q: ");
    mpz_out_str(stdout, 16, quot);
    printf("\n");
    printf("correct: ");
    mpz_out_str(stdout, 16, temp);
    printf("\n");
    fprintf(stderr, "barrett failed\n");
    exit(1);
  }

  mpz_set(q._z, quot);

  mpz_clear(temp);
  mpz_clear(quot);
  mpz_clear(rem);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::barrett_rem(Reg &q, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  mpz_t   temp, quot, rem;
  int32_t count;

  mpz_init(temp);
  mpz_init(quot);
  mpz_init(rem);

  // compute top
  mpz_fdiv_q_2exp(temp, num._z, bits-denom_clz);

  // q=(top * approx + top)>>bits
  mpz_mul(quot, temp, approx._z);
  mpz_fdiv_q_2exp(quot, quot, bits);
  mpz_add(quot, quot, temp);

  // q=min(q+3, beta-1), min might be unnecessary
  mpz_add_ui(quot, quot, 3);
  if(mpz_tstbit(quot, bits)==1) {
    mpz_set_ui(quot, 1);
    mpz_mul_2exp(quot, quot, bits);
    mpz_sub_ui(quot, quot, 1);
  }

  // r=num-q*denom
  mpz_mul(rem, quot, denom._z);
  mpz_sub(rem, num._z, rem);

  // correction steps
  count=0;
  while(count<3 && mpz_sgn(rem)==-1) {
    mpz_add(rem, rem, denom._z);
    count++;
  }

  // check it
  mpz_fdiv_r(temp, num._z, denom._z);
  if(mpz_cmp(rem, temp)!=0) {
    printf("    num: ");
    mpz_out_str(stdout, 16, num._z);
    printf("\n");
    printf("  denom: ");
    mpz_out_str(stdout, 16, denom._z);
    printf("\n");
    printf(" approx: ");
    mpz_out_str(stdout, 16, approx._z);
    printf("\n");
    printf("    rem: ");
    mpz_out_str(stdout, 16, rem);
    printf("\n");
    printf("      q: ");
    mpz_out_str(stdout, 16, quot);
    printf("\n");
    printf("correct: ");
    mpz_out_str(stdout, 16, temp);
    printf("\n");
    fprintf(stderr, "barrett failed\n");
    exit(1);
  }

  mpz_set(q._z, rem);

  mpz_clear(temp);
  mpz_clear(quot);
  mpz_clear(rem);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::barrett_div_rem(Reg &q, Reg &r, const Reg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  mpz_t   temp, quot, rem;
  int32_t count;

  mpz_init(temp);
  mpz_init(quot);
  mpz_init(rem);

  // compute top
  mpz_fdiv_q_2exp(temp, num._z, bits-denom_clz);

  // q=(top * approx + top)>>bits
  mpz_mul(quot, temp, approx._z);
  mpz_fdiv_q_2exp(quot, quot, bits);
  mpz_add(quot, quot, temp);

  // q=min(q+3, beta-1), min might be unnecessary
  mpz_add_ui(quot, quot, 3);
  if(mpz_tstbit(quot, bits)==1) {
    mpz_set_ui(quot, 1);
    mpz_mul_2exp(quot, quot, bits);
    mpz_sub_ui(quot, quot, 1);
  }

  // r=num-q*denom
  mpz_mul(rem, quot, denom._z);
  mpz_sub(rem, num._z, rem);

  // correction steps
  count=0;
  while(count<3 && mpz_sgn(rem)==-1) {
    mpz_sub_ui(quot, quot, 1);
    mpz_add(rem, rem, denom._z);
    count++;
  }

  // check it
  mpz_fdiv_q(temp, num._z, denom._z);
  if(mpz_cmp(quot, temp)!=0) {
    printf("    num: ");
    mpz_out_str(stdout, 16, num._z);
    printf("\n");
    printf("  denom: ");
    mpz_out_str(stdout, 16, denom._z);
    printf("\n");
    printf(" approx: ");
    mpz_out_str(stdout, 16, approx._z);
    printf("\n");
    printf("    rem: ");
    mpz_out_str(stdout, 16, rem);
    printf("\n");
    printf("      q: ");
    mpz_out_str(stdout, 16, quot);
    printf("\n");
    printf("correct: ");
    mpz_out_str(stdout, 16, temp);
    printf("\n");
    fprintf(stderr, "barrett failed\n");
    exit(1);
  }

  mpz_set(q._z, quot);
  mpz_set(r._z, rem);

  mpz_clear(temp);
  mpz_clear(quot);
  mpz_clear(rem);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::barrett_div_wide(Reg &q, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_cmp(num._high._z, denom._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }
  mpz_init(temp);
  mpz_set(temp, num._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, num._low._z);
  mpz_fdiv_q(q._z, temp, denom._z);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::barrett_rem_wide(Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_cmp(num._high._z, denom._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }
  mpz_init(temp);
  mpz_set(temp, num._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, num._low._z);
  mpz_fdiv_r(r._z, temp, denom._z);
  mpz_clear(temp);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::barrett_div_rem_wide(Reg &q, Reg &r, const WideReg &num, const Reg &denom, const Reg &approx, const uint32_t denom_clz) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_cmp(num._high._z, denom._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }
  mpz_init(temp);
  mpz_set(temp, num._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, num._low._z);
  mpz_fdiv_q(q._z, temp, denom._z);
  mpz_fdiv_r(r._z, temp, denom._z);
  mpz_clear(temp);
}


/*
template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::barrett_div_wide(Reg &q, const WideReg &num, const Reg &denom, const Reg &approx) const {
  mpz_t temp;

  if(_context.check_errors()) {
    if(mpz_cmp(num._high._z, denom._z)>=0) {
      _context.report_error(Error::kDivsionOverflow);
      return;
    }
  }
  mpz_init(temp);
  mpz_set(temp, num._high._z);
  mpz_mul_2exp(temp, temp, bits);
  mpz_add(temp, temp, num._low._z);
  mpz_fdiv_q(q._z, temp, denom._z);
  mpz_clear(temp);
}
*/

/* load and store routines */
template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::load(Reg &r, Mem<bits> *const address) const {
  mpz_import(r._z, (bits+31)/32, -1, sizeof(uint32_t), 0, 0, (uint32_t *)address);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::store(Mem<bits> *address, const Reg &a) const {
  size_t words;

  if(mpz_sizeinbase(a._z, 2)>bits) {
    fprintf(stderr, "from_mpz failed -- result does not fit");
    exit(1);
  }

  mpz_export((uint32_t *)address, &words, -1, sizeof(uint32_t), 0, 0, a._z);
  while(words<(bits+31)/32)
    ((uint32_t *)address)[words++]=0;
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::load_shorter(Reg &dst, uint32_t *const src, uint32_t mem_limb_count) const {
  constexpr uint32_t nlimb = (bits+31)/32;
  const uint32_t words_togo = nlimb < mem_limb_count ? nlimb : mem_limb_count;
  mpz_import(dst._z, words_togo , -1, sizeof(uint32_t), 0, 0, (uint32_t *)src);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::store_shorter(uint32_t *dst, const Reg &src, uint32_t mem_limb_count) const {
  size_t words;
  mpz_t ret;

  if(mpz_sizeinbase(src._z, 2)>bits) {
    fprintf(stderr, "from_mpz failed -- result does not fit");
    exit(1);
  }

  mp_bitcnt_t dst_bit_count = 32*mem_limb_count;
  mpz_init2(ret, dst_bit_count);
  mpz_fdiv_r_2exp(ret, src._z, dst_bit_count);
  mpz_export(dst, &words, -1, sizeof(uint32_t), 0, 0, ret);
  while(words<mem_limb_count)
    ((uint32_t *)dst)[words++]=0;
  mpz_clear(ret);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::load(Reg &r, LocalMem *const address) const {
  mpz_set(r._z, address->_z);
}

template<class context_t, uint32_t bits, ConvergenceKind convergence>
void GmpBnEnv<context_t, bits, convergence>::store(LocalMem *address, const Reg &a) const {
  mpz_set(address->_z, a._z);
}

} // namespace cgbn
