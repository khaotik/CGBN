#pragma once
#include <stdint.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <gmp.h> 

#include "cgbn/cgbn.h"
#include "types.h"

typedef enum test_enum {
  test_set_1, test_swap_1, test_add_1, test_negate_1, test_sub_1,
  test_mul_1, test_mul_high_1, test_sqr_1, test_sqr_high_1, test_div_1, test_rem_1,
  test_div_rem_1, test_sqrt_1, test_sqrt_rem_1, test_equals_1, test_equals_2, test_equals_3, test_compare_1, test_compare_2,
  test_compare_3, test_compare_4, test_extract_bits_1, test_insert_bits_1,
  
  test_all_set_ui32,
  test_get_ui32_set_ui32_1, test_add_ui32_1, test_sub_ui32_1, test_mul_ui32_1, test_div_ui32_1, test_rem_ui32_1, 
  test_all_equals_ui32_1, test_all_equals_ui32_2,
  test_equals_ui32_1, test_equals_ui32_2, test_equals_ui32_3, test_equals_ui32_4, test_compare_ui32_1, test_compare_ui32_2,
  test_extract_bits_ui32_1, test_insert_bits_ui32_1, test_binary_inverse_ui32_1, test_gcd_ui32_1,
  
  test_mul_wide_1, test_sqr_wide_1, test_div_wide_1, test_rem_wide_1, test_div_rem_wide_1, test_sqrt_wide_1, test_sqrt_rem_wide_1,
  
  test_bitwise_and_1, test_bitwise_ior_1, test_bitwise_xor_1, test_bitwise_complement_1, test_bitwise_select_1, test_bitwise_mask_copy_1,
  test_bitwise_mask_and_1, test_bitwise_mask_ior_1, test_bitwise_mask_xor_1, test_bitwise_mask_select_1, test_shift_left_1, 
  test_shift_right_1, test_rotate_left_1, test_rotate_right_1, test_pop_count_1, test_clz_1, test_ctz_1,
  
  test_accumulator_1, test_accumulator_2, test_binary_inverse_1, test_gcd_1, test_modular_inverse_1, test_modular_power_1,
  test_bn2mont_1, test_mont2bn_1, test_mont_mul_1, test_mont_sqr_1, test_mont_reduce_wide_1, test_barrett_div_1,
  test_barrett_rem_1, test_barrett_div_rem_1, test_barrett_div_wide_1, test_barrett_rem_wide_1, test_barrett_div_rem_wide_1
} test_t;

template<test_t test, bool is_gpu, class params>
struct implementation {
  // TODO.perf make it inline
  __host__ __device__ static void run(
      typename types<params>::input_t *inputs,
      typename types<params>::output_t *outputs, int32_t instance);
};


