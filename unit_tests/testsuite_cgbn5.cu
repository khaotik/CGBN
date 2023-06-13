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

REGISTER_TYPED_TEST_SUITE_P(CGBN5,
 accumulator_1, accumulator_2, binary_inverse_1, gcd_1, modular_inverse_1, modular_power_1,
 bn2mont_1, mont2bn_1, mont_mul_1, mont_sqr_1, mont_reduce_wide_1, barrett_div_1, barrett_rem_1,
 barrett_div_rem_1, barrett_div_wide_1, barrett_rem_wide_1, barrett_div_rem_wide_1
);
INSTANTIATE_TYPED_TEST_SUITE_P(ALL, CGBN5, test_sizes);
