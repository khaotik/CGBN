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

REGISTER_TYPED_TEST_SUITE_P(CGBN2, 
 all_set_ui32,
 get_ui32_set_ui32_1, add_ui32_1, sub_ui32_1, mul_ui32_1, div_ui32_1, rem_ui32_1, 
 all_equals_ui32_1, all_equals_ui32_2,
 equals_ui32_1, equals_ui32_2, equals_ui32_3, equals_ui32_4, compare_ui32_1, compare_ui32_2,
 extract_bits_ui32_1, insert_bits_ui32_1, binary_inverse_ui32_1, gcd_ui32_1
);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN2, test_sizes);
