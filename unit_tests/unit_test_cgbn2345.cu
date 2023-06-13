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
class CGBN2 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

template<class T>
class CGBN3 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

template<class T>
class CGBN4 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

template<class T>
class CGBN5 : public testing::Test {
  public:
  static const uint32_t TPB=T::TPB;
  static const uint32_t MAX_ROTATION=T::MAX_ROTATION;
  static const uint32_t SHM_LIMIT=T::SHM_LIMIT;
  static const bool     CONSTANT_TIME=T::CONSTANT_TIME;

  static const uint32_t BITS=T::BITS;
  static const uint32_t TPI=T::TPI;

  static const uint32_t size=T::BITS;
};

TYPED_TEST_SUITE_P(CGBN2);
TYPED_TEST_SUITE_P(CGBN3);
TYPED_TEST_SUITE_P(CGBN4);
TYPED_TEST_SUITE_P(CGBN5);

TYPED_TEST_P(CGBN2, all_set_ui32) {
  bool result=TestContext<test_all_set_ui32, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, get_ui32_set_ui32_1) {
  bool result=TestContext<test_get_ui32_set_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, add_ui32_1) {
  bool result=TestContext<test_add_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, sub_ui32_1) {
  bool result=TestContext<test_sub_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, mul_ui32_1) {
  bool result=TestContext<test_mul_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, div_ui32_1) {
  bool result=TestContext<test_div_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, rem_ui32_1) {
  bool result=TestContext<test_rem_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, all_equals_ui32_1) {
  bool result=TestContext<test_all_equals_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, all_equals_ui32_2) {
  bool result=TestContext<test_all_equals_ui32_2, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_1) {
  bool result=TestContext<test_equals_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_2) {
  bool result=TestContext<test_equals_ui32_2, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_3) {
  bool result=TestContext<test_equals_ui32_3, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, equals_ui32_4) {
  bool result=TestContext<test_equals_ui32_4, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, compare_ui32_1) {
  bool result=TestContext<test_compare_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, compare_ui32_2) {
  bool result=TestContext<test_compare_ui32_2, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, extract_bits_ui32_1) {
  bool result=TestContext<test_extract_bits_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, insert_bits_ui32_1) {
  bool result=TestContext<test_insert_bits_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, binary_inverse_ui32_1) {
  bool result=TestContext<test_binary_inverse_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN2, gcd_ui32_1) {
  bool result=TestContext<test_gcd_ui32_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, mul_wide_1) {
  bool result=TestContext<test_mul_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, sqr_wide_1) {
  bool result=TestContext<test_sqr_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, div_wide_1) {
  bool result=TestContext<test_div_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, rem_wide_1) {
  bool result=TestContext<test_rem_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, div_rem_wide_1) {
  bool result=TestContext<test_div_rem_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, sqrt_wide_1) {
  bool result=TestContext<test_sqrt_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN3, sqrt_rem_wide_1) {
  bool result=TestContext<test_sqrt_rem_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_and_1) {
  bool result=TestContext<test_bitwise_and_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_ior_1) {
  bool result=TestContext<test_bitwise_ior_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_xor_1) {
  bool result=TestContext<test_bitwise_xor_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}


TYPED_TEST_P(CGBN4, bitwise_complement_1) {
  bool result=TestContext<test_bitwise_complement_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_select_1) {
  bool result=TestContext<test_bitwise_select_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_copy_1) {
  bool result=TestContext<test_bitwise_mask_copy_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_and_1) {
  bool result=TestContext<test_bitwise_mask_and_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_ior_1) {
  bool result=TestContext<test_bitwise_mask_ior_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_xor_1) {
  bool result=TestContext<test_bitwise_mask_xor_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, bitwise_mask_select_1) {
  bool result=TestContext<test_bitwise_mask_select_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, shift_left_1) {
  bool result=TestContext<test_shift_left_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, shift_right_1) {
  bool result=TestContext<test_shift_right_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, rotate_left_1) {
  bool result=TestContext<test_rotate_left_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, rotate_right_1) {
  bool result=TestContext<test_rotate_right_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, pop_count_1) {
  bool result=TestContext<test_pop_count_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, clz_1) {
  bool result=TestContext<test_clz_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN4, ctz_1) {
  bool result=TestContext<test_ctz_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, accumulator_1) {
  bool result=TestContext<test_accumulator_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, accumulator_2) {
  bool result=TestContext<test_accumulator_2, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, binary_inverse_1) {
  bool result=TestContext<test_binary_inverse_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, gcd_1) {
  bool result=TestContext<test_gcd_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, modular_inverse_1) {
  bool result=TestContext<test_modular_inverse_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, modular_power_1) {
  bool result=TestContext<test_modular_power_1, TestFixture>(MEDIUM_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, bn2mont_1) {
  bool result=TestContext<test_bn2mont_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont2bn_1) {
  bool result=TestContext<test_mont2bn_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont_mul_1) {
  bool result=TestContext<test_mont_mul_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont_sqr_1) {
  bool result=TestContext<test_mont_sqr_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, mont_reduce_wide_1) {
  bool result=TestContext<test_mont_reduce_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_1) {
  bool result=TestContext<test_barrett_div_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_rem_1) {
  bool result=TestContext<test_barrett_rem_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_rem_1) {
  bool result=TestContext<test_barrett_div_rem_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_wide_1) {
  bool result=TestContext<test_barrett_div_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_rem_wide_1) {
  bool result=TestContext<test_barrett_rem_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

TYPED_TEST_P(CGBN5, barrett_div_rem_wide_1) {
  bool result=TestContext<test_barrett_div_rem_wide_1, TestFixture>(LONG_TEST)();
  EXPECT_TRUE(result);
}

REGISTER_TYPED_TEST_SUITE_P(CGBN2, 
 all_set_ui32,
 get_ui32_set_ui32_1, add_ui32_1, sub_ui32_1, mul_ui32_1, div_ui32_1, rem_ui32_1, 
 all_equals_ui32_1, all_equals_ui32_2,
 equals_ui32_1, equals_ui32_2, equals_ui32_3, equals_ui32_4, compare_ui32_1, compare_ui32_2,
 extract_bits_ui32_1, insert_bits_ui32_1, binary_inverse_ui32_1, gcd_ui32_1
);
REGISTER_TYPED_TEST_SUITE_P(CGBN3,
 mul_wide_1, sqr_wide_1, div_wide_1, rem_wide_1, div_rem_wide_1, sqrt_wide_1, sqrt_rem_wide_1
);
REGISTER_TYPED_TEST_SUITE_P(CGBN4,
 bitwise_and_1, bitwise_ior_1, bitwise_xor_1, bitwise_complement_1, bitwise_select_1, 
 bitwise_mask_copy_1, bitwise_mask_and_1, bitwise_mask_ior_1, bitwise_mask_xor_1, bitwise_mask_select_1,
 shift_left_1, shift_right_1, rotate_left_1, rotate_right_1, pop_count_1, clz_1, ctz_1
);
REGISTER_TYPED_TEST_SUITE_P(CGBN5,
 accumulator_1, accumulator_2, binary_inverse_1, gcd_1, modular_inverse_1, modular_power_1,
 bn2mont_1, mont2bn_1, mont_mul_1, mont_sqr_1, mont_reduce_wide_1, barrett_div_1, barrett_rem_1,
 barrett_div_rem_1, barrett_div_wide_1, barrett_rem_wide_1, barrett_div_rem_wide_1
);

INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN2, test_sizes);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN3, test_sizes);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN4, test_sizes);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN5, test_sizes);
