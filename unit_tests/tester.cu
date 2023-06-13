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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gmp.h>
#include "common.cuh"


/*
int main() {
  run_test<test_add_1, 2048>(LONG_TEST);
  run_test<test_sub_1, 2048>(LONG_TEST);
}
*/

int main(int argc, char **argv) {
  int nDevice=-1, result;
  
  cudaGetDeviceCount(&nDevice);

  if(nDevice<=0) {
    printf("Error no cuda device found.  Aborting tests\n");
    exit(EXIT_FAILURE);
  }
  
  testing::InitGoogleTest(&argc, argv);
  result=RUN_ALL_TESTS();
  // TODO use gtest to initialize test seed and report seed
  /*
  if(result!=0)
    printf("Please report random seed %08X along with failure\n", _seed);
    */
  return result;
}

