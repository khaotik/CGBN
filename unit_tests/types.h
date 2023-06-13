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
#pragma once
#include <type_traits>
#include "cgbn/cgbn.h"

template<typename params, bool is_gpu=false>
struct types {
  struct input_t {
    cgbn_mem_t<params::size> h1;
    cgbn_mem_t<params::size> h2;
    cgbn_mem_t<params::size> x1;
    cgbn_mem_t<params::size> x2;
    cgbn_mem_t<params::size> x3;
    // keep everything 128 byte aligned
    uint32_t                 u[32];
  };

  struct output_t {
    cgbn_mem_t<params::size> r1;
    cgbn_mem_t<params::size> r2;
  };

  using bn_context_t = std::conditional_t<is_gpu,
        cgbn_cuda_context_t<params::TPI>,
        cgbn_gmp_context_t<params::TPI> >;
  using bn_env_t = std::conditional_t<is_gpu,
    cgbn_cuda_env_t<bn_context_t, params::size>,
    cgbn_gmp_env_t<bn_context_t, params::size>
  >;
  using bn_t = typename bn_env_t::cgbn_t;
  using wide_t = typename bn_env_t::cgbn_wide_t;
  using bn_acc_t = typename bn_env_t::cgbn_accumulator_t;
};
