/***
__device__ inline
uint32_t fastRngI32(uint32_t x) {
  const uint32_t p = uint32_t(0)-5;
  uint32_t ret=1;
  #pragma unroll
  for (uint32_t i=(1<<31); i; i>>=1) {
    ret = (((uint64_t)ret) * ret) % p;
    if (i&x) {
      if (ret & (1<<31)) {
        ret = ((ret<<1)%p + 5)%p;
      } else {
        ret = (ret<<1)%p;
      }
    }
  }
  return ret;
}
***/
#include "testcase_common.cuh"

template<bool is_gpu, class params>
struct TestImpl<test_all_equals_ui32_1, is_gpu, params> {
  static const uint32_t TPI=params::TPI;
  static const uint32_t BITS=params::BITS;

  typedef cgbn_context_t<TPI, params, is_gpu>    context_t;
  typedef cgbn_env_t<context_t, BITS>    env_t;
  typedef typename env_t::cgbn_t         bn_t;

  public:
  __device__ __host__ static void run(
      typename TestTrait<params>::input_t *inputs,
      typename TestTrait<params>::output_t *outputs, int32_t instance) {
    // uint32_t x = fastRngI32(instance);
    context_t context(cgbn_print_monitor);
    env_t     env(context);
    constexpr int nword = BITS/32;
    bn_t x, ret;
    uint32_t u32=0;

    cgbn_load(env, x, &(inputs[instance].h1));
    u32 = cgbn_extract_bits_ui32(env, x, 0, 32);
    cgbn_set_ui32(env, x, u32);
    #pragma unroll
    for (int i=1; i<nword; ++i) {
      cgbn_shift_left(env, x, x, 32);
      cgbn_add_ui32(env, x, x, u32);
    }
    bool equal = cgbn_all_equals_ui32(env, x, u32);
    cgbn_set_ui32(env, ret, (equal==false) ? 0 : 1);
    cgbn_store(env, &(outputs[instance].r1), ret);
  }
};

TYPED_TEST_P(CGBN2, all_equals_ui32_1) {
  bool result=TestContext<test_all_equals_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);}