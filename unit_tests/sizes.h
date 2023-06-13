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
#include <stdint.h>
#include "testcase_common.cuh"

struct size32t4 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=32;
  static constexpr uint32_t TPI=4;
};

struct size64t4 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=64;
  static constexpr uint32_t TPI=4;
};

struct size96t4 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=96;
  static constexpr uint32_t TPI=4;
};

struct size128t4 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=128;
  static constexpr uint32_t TPI=4;
};

struct size192t8 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=192;
  static constexpr uint32_t TPI=8;
};

struct size256t8 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;   // SHM_LIMIT or SHARED_LIMIT
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=256;
  static constexpr uint32_t TPI=8;
};

struct size288t8 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=288;
  static constexpr uint32_t TPI=8;
};

struct size512t8 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=512;
  static constexpr uint32_t TPI=8;
};

struct size1024t8 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=1024;
  static constexpr uint32_t TPI=8;
};

struct size1024t16 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=1024;
  static constexpr uint32_t TPI=16;
};

struct size1024t32 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=1024;
  static constexpr uint32_t TPI=32;
};

struct size2048t32 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=2048;
  static constexpr uint32_t TPI=32;
};

struct size3072t32 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=3072;
  static constexpr uint32_t TPI=32;
};

struct size4096t32 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=4096;
  static constexpr uint32_t TPI=32;
};

struct size8192t32 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=8192;
  static constexpr uint32_t TPI=32;
};

struct size16384t32 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=16384;
  static constexpr uint32_t TPI=32;
};

struct size32768t32 {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;

  static constexpr uint32_t BITS=32768;
  static constexpr uint32_t TPI=32;
};

#ifdef FULL_TEST
using test_sizes = testing::Types<
  size32t4,
  size64t4,
  size96t4,
  size128t4,
  size192t8,
  size256t8,
  size288t8,
  size512t8,
  size1024t8,
  size1024t16,
  size1024t32,
  size2048t32,
  size3072t32,
  size4096t32,
  size8192t32,
  size16384t32,
  size32768t32,
>;
#else
using test_sizes = testing::Types<
  size32t4,
  size128t4,
  size192t8,
  size256t8,
  size288t8,
  size512t8,
  size1024t8,
  size2048t32,
  size3072t32,
  size4096t32,
  size8192t32,
>;
#endif
