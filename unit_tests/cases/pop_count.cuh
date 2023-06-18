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
struct TestImpl<test_pop_count_1, is_gpu, params> {
  static const uint32_t TPI=params::TPI;
  static const uint32_t BITS=params::BITS;

  typedef cgbn::BnContext<TPI, params, is_gpu>    context_t;
  typedef cgbn::BnEnv<context_t, BITS>    env_t;
  typedef typename env_t::Reg         bn_t;

  public:
  __device__ __host__ static void run(typename TestTrait<params>::input_t *inputs, typename TestTrait<params>::output_t *outputs, int32_t instance) {
    context_t context(cgbn::MonitorKind::kPrint);
    env_t     env(context);
    bn_t      h1, r1;

    cgbn::load(env, h1, &(inputs[instance].h1));

    cgbn::set_ui32(env, r1, cgbn::pop_count(env, h1));

    cgbn::store(env, &(outputs[instance].r1), r1);
  }
};

TYPED_TEST_P(CGBN4, pop_count_1) {
  bool result=TestContext<test_pop_count_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);}