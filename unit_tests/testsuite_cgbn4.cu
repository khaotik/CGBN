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

REGISTER_TYPED_TEST_SUITE_P(CGBN4,
 bitwise_and_1, bitwise_ior_1, bitwise_xor_1, bitwise_complement_1, bitwise_select_1, 
 bitwise_mask_copy_1, bitwise_mask_and_1, bitwise_mask_ior_1, bitwise_mask_xor_1, bitwise_mask_select_1,
 shift_left_1, shift_right_1, rotate_left_1, rotate_right_1, pop_count_1, clz_1, ctz_1
);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN4, test_sizes);
