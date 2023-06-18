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
#include "testcase_common.cuh"

template<bool is_gpu, class params>
struct TestImpl<test_modular_power_1, is_gpu, params> {
  static const uint32_t TPI=params::TPI;
  static const uint32_t BITS=params::BITS;

  typedef cgbn::BnContext<TPI, params, is_gpu>    context_t;
  typedef cgbn::BnEnv<context_t, BITS>    env_t;
  typedef typename env_t::Reg         bn_t;

  public:
  __device__ __host__ static void run(typename TestTrait<params>::input_t *inputs, typename TestTrait<params>::output_t *outputs, int32_t instance) {
    context_t context(cgbn::MonitorKind::kPrint);
    env_t     env(context);
    bn_t      h1, h2, x1, r1, temp;
    int32_t   compare;

    cgbn::load(env, h1, &(inputs[instance].h1));
    cgbn::load(env, h2, &(inputs[instance].h2));
    cgbn::load(env, x1, &(inputs[instance].x1));

    compare=cgbn::compare(env, h1, h2);
    if(compare>0) {
      cgbn::set(env, temp, h1);
      cgbn::set(env, h1, h2);
      cgbn::set(env, h2, temp);
    }
    else if(compare==0)
      cgbn::set_ui32(env, h1, 0);

    cgbn::bitwise_mask_and(env, x1, x1, 512);

    if(!cgbn::equals_ui32(env, h2, 0)) {
      cgbn::modular_power(env, r1, h1, x1, h2);
      cgbn::store(env, &(outputs[instance].r1), r1);
    }
  }
};

TYPED_TEST_P(CGBN5, modular_power_1) {
  bool result=TestContext<test_modular_power_1, TestFixture>(MEDIUM_TEST)();
  EXPECT_TRUE(result);}