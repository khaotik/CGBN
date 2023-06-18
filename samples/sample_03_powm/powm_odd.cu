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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.cuh"
#include "../utility/support.h"

// For this example, there are quite a few template parameters that are used to generate the actual code.
// In order to simplify passing many parameters, we use the same approach as the CGBN library, which is to
// create a container class with static constexprants and then pass the class.

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
//   WINDOW_BITS     - number of bits to use for the windowed exponentiation

template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class powm_params_t {
  public:
  // parameters used by the CGBN context
  static constexpr uint32_t TPB=0;                     // get TPB from blockDim.x  
  static constexpr uint32_t MAX_ROTATION=4;            // good default value
  static constexpr uint32_t SHM_LIMIT=0;               // no shared mem available
  static constexpr bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet
  
  // parameters used locally in the application
  static constexpr uint32_t TPI=tpi;                   // threads per instance
  static constexpr uint32_t BITS=bits;                 // instance size
  static constexpr uint32_t WINDOW_BITS=window_bits;   // window size
};

template<class params>
class powm_odd_t {
  public:
  static constexpr uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable
  using Mem = cgbn::Mem<params::BITS>;
  
  // define the instance structure
  typedef struct {
    Mem x;
    Mem power;
    Mem modulus;
    Mem result;
  } instance_t;

  using context_t = cgbn::BnContext<params::TPI, params>;
  using env_t = cgbn::BnEnv<context_t, params::BITS>;
  using bn_t = typename env_t::Reg;
  using bn_local_t = typename env_t::LocalMem;

  context_t _context;
  env_t     _env;
  int32_t   _instance;
  
  __device__ __forceinline__ powm_odd_t(
      cgbn::MonitorKind monitor,
      cgbn::ErrorReport *report,
      int32_t instance
    ):
    _context(monitor, report, (uint32_t)instance),
    _env(_context),
    _instance(instance) {}

  __device__ __forceinline__ void fixed_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t       t;
    bn_local_t window[1<<window_bits];
    int32_t    index, position, offset;
    uint32_t   np0;

    // conmpute x^power mod modulus, using the fixed window algorithm
    // requires:  x<modulus,  modulus is odd

    // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
    cgbn::negate(_env, t, modulus);
    cgbn::store(_env, window+0, t);
    
    // convert x into Montgomery space, store into window table
    np0=cgbn::bn2mont(_env, result, x, modulus);
    cgbn::store(_env, window+1, result);
    cgbn::set(_env, t, result);
    
    // compute x^2, x^3, ... x^(2^window_bits-1), store into window table
    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      cgbn::mont_mul(_env, result, result, t, modulus, np0);
      cgbn::store(_env, window+index, result);
    }

    // find leading high bit
    position=params::BITS - cgbn::clz(_env, power);

    // break the exponent into chunks, each window_bits in length
    // load the most significant non-zero exponent chunk
    offset=position % window_bits;
    if(offset==0)
      position=position-window_bits;
    else
      position=position-offset;
    index=cgbn::extract_bits_ui32(_env, power, position, window_bits);
    cgbn::load(_env, result, window+index);
    
    // process the remaining exponent chunks
    while(position>0) {
      // square the result window_bits times
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
        cgbn::mont_sqr(_env, result, result, modulus, np0);
      
      // multiply by next exponent chunk
      position=position-window_bits;
      index=cgbn::extract_bits_ui32(_env, power, position, window_bits);
      cgbn::load(_env, t, window+index);
      cgbn::mont_mul(_env, result, result, t, modulus, np0);
    }
    
    // we've processed the exponent now, convert back to normal space
    cgbn::mont2bn(_env, result, result, modulus, np0);
  }
  
  __device__ __forceinline__ void sliding_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t         t, starts;
    int32_t      index, position, leading;
    uint32_t     mont_inv;
    bn_local_t   odd_powers[1<<window_bits-1];

    // conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
    // requires:  x<modulus,  modulus is odd
        
    // find the leading one in the power
    leading=params::BITS-1-cgbn::clz(_env, power);
    if(leading>=0) {
      // convert x into Montgomery space, store in the odd powers table
      mont_inv=cgbn::bn2mont(_env, result, x, modulus);
      
      // compute t=x^2 mod modulus
      cgbn::mont_sqr(_env, t, result, modulus, mont_inv);
      
      // compute odd powers window table: x^1, x^3, x^5, ...
      cgbn::store(_env, odd_powers, result);
      #pragma nounroll
      for(index=1;index<(1<<window_bits-1);index++) {
        cgbn::mont_mul(_env, result, result, t, modulus, mont_inv);
        cgbn::store(_env, odd_powers+index, result);
      }
  
      // starts contains an array of bits indicating the start of a window
      cgbn::set_ui32(_env, starts, 0);
  
      // organize p as a sequence of odd window indexes
      position=0;
      while(true) {
        if(cgbn::extract_bits_ui32(_env, power, position, 1)==0)
          position++;
        else {
          cgbn::insert_bits_ui32(_env, starts, starts, position, 1, 1);
          if(position+window_bits>leading)
            break;
          position=position+window_bits;
        }
      }
  
      // load first window.  Note, since the window index must be odd, we have to
      // divide it by two before indexing the window table.  Instead, we just don't
      // load the index LSB from power
      index=cgbn::extract_bits_ui32(_env, power, position+1, window_bits-1);
      cgbn::load(_env, result, odd_powers+index);
      position--;
      
      // Process remaining windows 
      while(position>=0) {
        cgbn::mont_sqr(_env, result, result, modulus, mont_inv);
        if(cgbn::extract_bits_ui32(_env, starts, position, 1)==1) {
          // found a window, load the index
          index=cgbn::extract_bits_ui32(_env, power, position+1, window_bits-1);
          cgbn::load(_env, t, odd_powers+index);
          cgbn::mont_mul(_env, result, result, t, modulus, mont_inv);
        }
        position--;
      }
      
      // convert result from Montgomery space
      cgbn::mont2bn(_env, result, result, modulus, mont_inv);
    }
    else {
      // p=0, thus x^p mod modulus=1
      cgbn::set_ui32(_env, result, 1);
    }
  }
  
  __host__ static instance_t *generate_instances(uint32_t count) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
    int         index;
  
    for(index=0;index<count;index++) {
      random_words(instances[index].x._limbs, params::BITS/32);
      random_words(instances[index].power._limbs, params::BITS/32);
      random_words(instances[index].modulus._limbs, params::BITS/32);

      // ensure modulus is odd
      instances[index].modulus._limbs[0] |= 1;

      // ensure modulus is greater than 
      if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)>0) {
        swap_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32);
        
        // modulus might now be even, ensure it's odd
        instances[index].modulus._limbs[0] |= 1;
      }
      else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)==0) {
        // since modulus is odd and modulus = x, we can just subtract 1 from x
        instances[index].x._limbs[0] -= 1;
      }
    }
    return instances;
  }
  
  __host__ static void verify_results(instance_t *instances, uint32_t count) {
    mpz_t x, p, m, computed, correct;
    
    mpz_init(x);
    mpz_init(p);
    mpz_init(m);
    mpz_init(computed);
    mpz_init(correct);
    
    for(int index=0;index<count;index++) {
      to_mpz(x, instances[index].x._limbs, params::BITS/32);
      to_mpz(p, instances[index].power._limbs, params::BITS/32);
      to_mpz(m, instances[index].modulus._limbs, params::BITS/32);
      to_mpz(computed, instances[index].result._limbs, params::BITS/32);
      
      mpz_powm(correct, x, p, m);
      if(mpz_cmp(correct, computed)!=0) {
        printf("gpu inverse kernel failed on instance %d\n", index);
        return;
      }
    }
  
    mpz_clear(x);
    mpz_clear(p);
    mpz_clear(m);
    mpz_clear(computed);
    mpz_clear(correct);
    
    printf("All results match\n");
  }
};

// kernel implementation using cgbn
// 
// Unfortunately, the kernel must be separate from the powm_odd_t class

template<class params>
__global__ void kernel_powm_odd(cgbn::ErrorReport *report, typename powm_odd_t<params>::instance_t *instances, uint32_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  powm_odd_t<params>                 po(cgbn::MonitorKind::kReport, report, instance);
  typename powm_odd_t<params>::bn_t  r, x, p, m;
  
  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn::load(po._env, x, &(instances[instance].x));
  cgbn::load(po._env, p, &(instances[instance].power));
  cgbn::load(po._env, m, &(instances[instance].modulus));
  
  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.  
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  po.fixed_window_powm_odd(r, x, p, m);
  //   OR
  // po.sliding_window_powm_odd(r, x, p, m);
  
  cgbn::store(po._env, &(instances[instance].result), r);
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef typename powm_odd_t<params>::instance_t instance_t;

  instance_t          *instances, *gpuInstances;
  cgbn::ErrorReport *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block
  
  printf("Genereating instances ...\n");
  instances=powm_odd_t<params>::generate_instances(instance_count);
  
  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn::cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");
  
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInstances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));
  
  printf("Verifying the results ...\n");
  powm_odd_t<params>::verify_results(instances, instance_count);
  
  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn::cgbn_error_report_free(report));
}

int main() {
  typedef powm_params_t<8, 1024, 5> params;
  
  run_test<params>(10000);
}
