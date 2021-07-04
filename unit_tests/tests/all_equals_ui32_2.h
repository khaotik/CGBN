/*
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
*/

template<class params>
struct implementation<test_all_equals_ui32_2, params> {
  static const uint32_t TPI=params::TPI;
  static const uint32_t BITS=params::BITS;

  typedef cgbn_context_t<TPI, params>    context_t;
  typedef cgbn_env_t<context_t, BITS>    env_t;
  typedef typename env_t::cgbn_t         bn_t;

  public:
  __device__ __host__ static void run(
      typename types<params>::input_t *inputs,
      typename types<params>::output_t *outputs, int32_t instance) {
    // uint32_t x = fastRngI32(instance);
    context_t context(cgbn_print_monitor);
    env_t     env(context);
    bn_t x, ret;
    uint32_t u32=0;

    cgbn_load(env, x, &(inputs[instance].h1));
    u32 = cgbn_extract_bits_ui32(env, x, 0, 32);
    bool equal = cgbn_all_equals_ui32(env, x, u32);
    cgbn_set_ui32(env, ret, (equal==false) ? 0 : 1);
    cgbn_store(env, &(outputs[instance].r1), ret);
  }
};
