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
#include "common.cuh"
#include "all_cases.cuh"

template<class T>
struct CGBN1 : public testing::Test {
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;
  
  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};
TYPED_TEST_SUITE_P(CGBN1);

// trying to hack
// #include "tests/set.cu"

TYPED_TEST_P(CGBN1, set_1) {
  bool result=TestContext<test_set_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, swap_1) {
  bool result=TestContext<test_swap_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, add_1) {
  bool result=TestContext<test_add_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sub_1) {
  bool result=TestContext<test_sub_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, negate_1) {
  bool result=TestContext<test_negate_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, mul_1) {
  bool result=TestContext<test_mul_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, mul_high_1) {
  bool result=TestContext<test_mul_high_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqr_1) {
  bool result=TestContext<test_sqr_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqr_high_1) {
  bool result=TestContext<test_sqr_high_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, div_1) {
  bool result=TestContext<test_div_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, rem_1) {
  bool result=TestContext<test_rem_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, div_rem_1) {
  bool result=TestContext<test_div_rem_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqrt_1) {
  bool result=TestContext<test_sqrt_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, sqrt_rem_1) {
  bool result=TestContext<test_sqrt_rem_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, equals_1) {
  bool result=TestContext<test_equals_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, equals_2) {
  bool result=TestContext<test_equals_2, TestFixture>(TINY_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, equals_3) {
  bool result=TestContext<test_equals_3, TestFixture>(TINY_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_1) {
  bool result=TestContext<test_compare_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_2) {
  bool result=TestContext<test_compare_2, TestFixture>(TINY_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_3) {
  bool result=TestContext<test_compare_3, TestFixture>(TINY_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, compare_4) {
  bool result=TestContext<test_compare_4, TestFixture>(TINY_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, extract_bits_1) {
  bool result=TestContext<test_extract_bits_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN1, insert_bits_1) {
  bool result=TestContext<test_insert_bits_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

REGISTER_TYPED_TEST_SUITE_P(CGBN1,
 set_1, swap_1, add_1, sub_1, negate_1,
 mul_1, mul_high_1, sqr_1, sqr_high_1, div_1, rem_1, div_rem_1, sqrt_1,
 sqrt_rem_1, equals_1, equals_2, equals_3, compare_1, compare_2, compare_3, compare_4,
 extract_bits_1, insert_bits_1
);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN1, test_sizes);
