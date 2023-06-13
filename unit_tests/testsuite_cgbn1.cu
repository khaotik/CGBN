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

REGISTER_TYPED_TEST_SUITE_P(CGBN1,
 set_1, swap_1, add_1, sub_1, negate_1,
 mul_1, mul_high_1, sqr_1, sqr_high_1, div_1, rem_1, div_rem_1, sqrt_1,
 sqrt_rem_1, equals_1, equals_2, equals_3, compare_1, compare_2, compare_3, compare_4,
 extract_bits_1, insert_bits_1
);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN1, test_sizes);
