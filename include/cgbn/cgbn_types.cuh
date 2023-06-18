#pragma once

namespace cgbn {

enum class Error: uint32_t {
  kSuccess=0,
  kUnsupportedTPI=1,
  kUnsupportedSize=2,
  kUnsupportedLimbsPerThread=3,
  kUnsupportedOperation=4,
  kTBPMismatch=5,
  kTPIMismatch=6,
  kDivisionByZero=7,
  kDivsionOverflow=8,
  kMontgomeryModulusError=9,
  kModulusNotOdd=10,
  kInversionDoesNotExist=11,
};
struct ErrorReport {
  volatile Error _error;
  uint32_t       _instance;
  dim3           _threadIdx;
  dim3           _blockIdx;
};
enum class MonitorKind: uint8_t {
  kNone = 0,       /* disable error checking - improves performance */
  kReport = 1,  /* writes errors to the reporter object, no other actions */
  kPrint = 2,   /* writes errors to the reporter and prints the error to stdout */
  kHalt = 15,    /* writes errors to the reporter and halts */
};
template<uint32_t bits>
struct Mem {
  uint32_t _limbs[(bits+31)/32];
};

template<uint32_t tpi, typename params, bool is_gpu> 
struct _ContextInfer;
template<uint32_t tpi, typename params, bool is_gpu=true> 
using BnContext = typename _ContextInfer<tpi, params, is_gpu>::type;

struct CgbnParam {
  uint32_t TPI;
};

/*
struct cgbn_cuda_default_parameters_t {
  static constexpr uint32_t TPB=0;
  static constexpr uint32_t MAX_ROTATION=4;
  static constexpr uint32_t SHM_LIMIT=0;
  static constexpr bool     CONSTANT_TIME=false;
};
struct cgbn_default_gmp_parameters_t {
  static constexpr uint32_t TPB=0;
};
*/

} // namespace cgbn
