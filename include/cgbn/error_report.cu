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

namespace cgbn {
inline cudaError_t cgbn_error_report_alloc(ErrorReport **report) {
  cudaError_t status;

  status=cudaMallocManaged((void **)report, sizeof(ErrorReport));
  if(status!=0)
    return status;
  (*report)->_error=Error::kSuccess;
  (*report)->_instance=0xFFFFFFFFu;
  (*report)->_threadIdx.x=0xFFFFFFFFu;
  (*report)->_threadIdx.y=0xFFFFFFFFu;
  (*report)->_threadIdx.z=0xFFFFFFFFu;
  (*report)->_blockIdx.x=0xFFFFFFFFu;
  (*report)->_blockIdx.y=0xFFFFFFFFu;
  (*report)->_blockIdx.z=0xFFFFFFFFu;
  return status;
}

inline cudaError_t cgbn_error_report_free(ErrorReport *report) {
  return cudaFree(report);
}

inline bool cgbn_error_report_check(ErrorReport *report) {
  return report->_error!=Error::kSuccess;
}

inline void cgbn_error_report_reset(ErrorReport *report) {
  report->_error=Error::kSuccess;
  report->_instance=0xFFFFFFFFu;
  report->_threadIdx.x=0xFFFFFFFFu;
  report->_threadIdx.y=0xFFFFFFFFu;
  report->_threadIdx.z=0xFFFFFFFFu;
  report->_blockIdx.x=0xFFFFFFFFu;
  report->_blockIdx.y=0xFFFFFFFFu;
  report->_blockIdx.z=0xFFFFFFFFu;
}

inline const char *cgbn_error_string(ErrorReport *report) {
  if(report->_error==Error::kSuccess)
    return NULL;
  switch(report->_error) {
    case Error::kUnsupportedTPI:
      return "unsupported threads per instance";
    case Error::kUnsupportedSize:
      return "unsupported size";
    case Error::kUnsupportedLimbsPerThread:
      return "unsupported limbs per thread";
    case Error::kUnsupportedOperation:
      return "unsupported operation";
    case Error::kTBPMismatch:
      return "TPB does not match blockDim.x";
    case Error::kTPIMismatch:
      return "TPI does not match env_t::TPI";
    case Error::kDivisionByZero:
      return "division by zero";
    case Error::kDivsionOverflow:
      return "division overflow";
    case Error::kMontgomeryModulusError:
      return "invalid montgomery modulus";
    case Error::kModulusNotOdd:
      return "invalid modulus (it must be odd)";
    case Error::kInversionDoesNotExist:
      return "inverse does not exist";      
    case Error::kSuccess:
      break;
  }
  return NULL;
}
}
