#pragma once
#include <gmp.h>

#include "sizes.h"
#include "testcase_common.cuh"

inline cudaError_t _checkCudaError(cudaError_t err, int line, const char* file) {
  if (err != cudaSuccess) {
    printf("%s at %s:%d: %s\n", cudaGetErrorName(err), file, line, cudaGetErrorString(err));
  } return err; }

#define $CUDA_CHECK(expr) do { if(cudaSuccess!=_checkCudaError(expr,__LINE__,__FILE__)) exit(123); } while (false)

constexpr int LONG_TEST   = 1000000;
constexpr int MEDIUM_TEST = 100000;
constexpr int SHORT_TEST  = 10000;
constexpr int TINY_TEST   = 1000;
constexpr int SINGLE_TEST = 1;

inline void
copy_words(uint32_t *from, uint32_t *to, uint32_t count) {
  int index;
  for(index=0;index<count;index++)
    to[index]=from[index];
}
inline void
print_words(uint32_t *x, uint32_t count) {
  int index;
  for(index=count-1;index>=0;index--)
    printf("%08X", x[index]);
  printf("\n"); }
inline int
compare_words(uint32_t *x, uint32_t *y, uint32_t count) {
  int index;
  for(index=count-1;index>=0;index--) {
    if(x[index]!=y[index]) {
      if(x[index]>y[index])
        return 1;
      else
        return -1;
    }
  }
  return 0;
}
inline void
random_words(uint32_t *x, uint32_t count, gmp_randstate_t state) {
  int32_t index;

  for(index=0;index<count;index++)
    x[index]=gmp_urandomb_ui(state, 32);
}
inline void
zero_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    x[index]=0;
}
inline void
hard_random_words(uint32_t *x, uint32_t count, gmp_randstate_t state) {
  uint32_t values[6]={0x0, 0x1, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF};
  int32_t  offset, bit, bits, index;

  switch(gmp_urandomb_ui(state, 16)%3) {
    case 0:
      for(index=0;index<count;index++)
        x[index]=gmp_urandomb_ui(state, 32);
      break;
    case 1:
      for(index=0;index<count;index++)
        x[index]=values[gmp_urandomb_ui(state, 16)%6];
      break;
    case 2:
      zero_words(x, count);
      offset=0;
      while(offset<count*32) {
        bit=gmp_urandomb_ui(state, 16)%2;
        bits=gmp_urandomb_ui(state, 32)%(32*count/2)+16;
        if(bit==1) {
          if(bits>count*32-offset)
            bits=count*32-offset;
          while(bits>0) {
            if(offset%32==0 && bits>=32) {
              while(bits>=32) {
                x[offset/32]=0xFFFFFFFF;
                bits-=32;
                offset+=32;
              }
            }
            else {
              x[offset/32]=x[offset/32] + (1<<offset%32);
              bits--;
              offset++;
            }
          }
        }
        else
          offset+=bits;
      }
      break;
  }
}

template<TestId test_id, typename test_param, typename input_ty, typename output_ty>
__global__ void
gpu_kernel(input_ty *inputs, output_ty *outputs, uint32_t count) {
  TestImpl<test_id, true, test_param> impl;
  int32_t inst_id=(blockIdx.x * blockDim.x + threadIdx.x)/test_param::TPI;
  if(inst_id>=count) return;
  impl.run(inputs, outputs, inst_id); }

template<TestId test_id, typename test_param>
class TestContext {
  using input_ty = typename TestTrait<test_param>::input_t;
  using output_ty =typename TestTrait<test_param>::output_t;
  gmp_randstate_t  _state;
  uint32_t         _seed=0;
  static constexpr
    uint32_t _bits=test_param::size;
  uint32_t         _count=0;
  void*            _cpu_data=nullptr;
  void*            _gpu_data=nullptr;

  void generate_data() {
    input_ty *inputs;
    int32_t instance;
      
    // printf("generating %d\n", params::size);
    free(_cpu_data); _cpu_data = nullptr;
    $CUDA_CHECK(cudaFree(_gpu_data)); _gpu_data=nullptr;
    _cpu_data=malloc(sizeof(input_ty)*_count);
      
    inputs=reinterpret_cast<input_ty*>(_cpu_data);
    gmp_randseed_ui(_state, _seed);    
    for(instance=0; instance<_count; instance++) {
      hard_random_words(inputs[instance].h1._limbs, test_param::size/32, _state);
      hard_random_words(inputs[instance].h2._limbs, test_param::size/32, _state);
      random_words(inputs[instance].x1._limbs, test_param::size/32, _state);
      random_words(inputs[instance].x2._limbs, test_param::size/32, _state);
      random_words(inputs[instance].x3._limbs, test_param::size/32, _state);
      random_words(inputs[instance].u, 32, _state); }
    $CUDA_CHECK(cudaMalloc((void **)&_gpu_data, sizeof(input_ty)*_count));
    $CUDA_CHECK(cudaMemcpy(_gpu_data, _cpu_data, sizeof(input_ty)*_count, cudaMemcpyHostToDevice));
  }
  /*
  input_ty*
  cpu_data() {
    if(_seed==0) {
      _seed=time(nullptr);
      gmp_randinit_default(_state); }
    generate_data();
    return reinterpret_cast<input_ty*>(_cpu_data); }
  input_ty*
  gpu_data() {
    if(_seed==0) {
      _seed=time(nullptr);
      gmp_randinit_default(_state); }
    generate_data();
    return reinterpret_cast<input_ty*>(_gpu_data);
  }
  */
  void gpu_run(input_ty *inputs, output_ty *outputs) {
    uint32_t TPB=(test_param::TPB==0) ? 128 : test_param::TPB;
    uint32_t TPI=test_param::TPI, IPB=TPB/TPI;
    uint32_t blocks=(_count+IPB+1)/IPB;
    gpu_kernel<test_id, test_param><<<blocks, TPB>>>(inputs, outputs, _count);
    $CUDA_CHECK(cudaGetLastError());
  }
  void cpu_run(input_ty *inputs, output_ty *outputs) {
    TestImpl<test_id, false, test_param> impl;
    #pragma omp parallel for
    for(int index=0; index<_count; index++)  {
      impl.run(inputs, outputs, index); }
  }

public:
  TestContext(uint32_t count) {
    _count = test_param::size>1024 ?
      count*(1024*1024/test_param::size)/1024 :
      count;
  }
  ~TestContext() {
    if (_cpu_data) free(_cpu_data);
    if (_gpu_data) $CUDA_CHECK(cudaFree(_gpu_data));
  }
  bool operator ()() {
    input_ty  *cpu_inputs, *gpu_inputs;
    output_ty *compare, *cpu_outputs, *gpu_outputs;
    int instance;
    _seed=time(nullptr);
    gmp_randinit_default(_state);
    generate_data();
    cpu_inputs = reinterpret_cast<input_ty*>(_cpu_data);
    gpu_inputs = reinterpret_cast<input_ty*>(_gpu_data);
    
    const auto out_memsize = sizeof(output_ty)*_count;
    compare=reinterpret_cast<output_ty*>(malloc(out_memsize));
    cpu_outputs=reinterpret_cast<output_ty*>(malloc(out_memsize));

    memset(cpu_outputs, 0, out_memsize);
    $CUDA_CHECK(cudaMalloc((void **)&gpu_outputs, out_memsize));
    $CUDA_CHECK(cudaMemset(gpu_outputs, 0, out_memsize));
    
    cpu_run(cpu_inputs, cpu_outputs);
    gpu_run(gpu_inputs, gpu_outputs);
    $CUDA_CHECK(cudaMemcpy(compare, gpu_outputs, out_memsize, cudaMemcpyDeviceToHost));
    
    for(instance=0;instance<_count;instance++) {
      if(compare_words(cpu_outputs[instance].r1._limbs, compare[instance].r1._limbs, test_param::size/32)!=0 || 
         compare_words(cpu_outputs[instance].r2._limbs, compare[instance].r2._limbs, test_param::size/32)!=0) {
        printf("Test failed at index %d\n", instance);
        printf("h1: ");
        print_words(cpu_inputs[instance].h1._limbs, test_param::size/32);
        printf("\n");
        printf("h2: ");
        print_words(cpu_inputs[instance].h2._limbs, test_param::size/32);
        printf("\n");
        printf("x1: ");
        print_words(cpu_inputs[instance].x1._limbs, test_param::size/32);
        printf("\n");
   //     printf("x2: ");
   //     print_words(cpu_inputs[instance].x2._limbs, test_param::size/32);
   //     printf("\n");
   //     printf("x3: ");
   //     print_words(cpu_inputs[instance].x3._limbs, test_param::size/32);
   //     printf("\n");
        printf("u0: %08X   u1: %08X   u2: %08X\n\n", cpu_inputs[instance].u[0], cpu_inputs[instance].u[1], cpu_inputs[instance].u[2]);
        printf("CPU R1: ");
        print_words(cpu_outputs[instance].r1._limbs, test_param::size/32);
        printf("\n");
        printf("GPU R1: ");
        print_words(compare[instance].r1._limbs, test_param::size/32);
        printf("\n");
        printf("CPU R2: ");
        print_words(cpu_outputs[instance].r2._limbs, test_param::size/32);
        printf("\n");
        printf("GPU R2: ");
        print_words(compare[instance].r2._limbs, test_param::size/32);
        printf("\n");
        return false;
      }
    }
    free(compare);
    free(cpu_outputs);
    $CUDA_CHECK(cudaFree(gpu_outputs));
    return true;
  }
};
