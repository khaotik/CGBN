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
struct TestImpl<test_insert_bits_ui32_1, is_gpu, params> {
  static const uint32_t TPI=params::TPI;
  static const uint32_t BITS=params::BITS;

  typedef cgbn::BnContext<TPI, params, is_gpu>    context_t;
  typedef cgbn::BnEnv<context_t, BITS>    env_t;
  typedef typename env_t::Reg         bn_t;

  public:
  __device__ __host__ static void run(typename TestTrait<params>::input_t *inputs, typename TestTrait<params>::output_t *outputs, int32_t instance) {
    context_t context(cgbn::MonitorKind::kPrint);
    env_t     env(context);
    bn_t      x1, r1;
    uint32_t  u1, u2, u3;

    cgbn::load(env, x1, &(inputs[instance].x1));
    u1=inputs[instance].u[0];
    u2=inputs[instance].u[1];
    u3=inputs[instance].u[2];

    cgbn::insert_bits_ui32(env, r1, x1, u1%BITS, u2%40, u3);

    cgbn::store(env, &(outputs[instance].r1), r1);
  }
};

TYPED_TEST_P(CGBN2, insert_bits_ui32_1) {
  bool result=TestContext<test_insert_bits_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);}