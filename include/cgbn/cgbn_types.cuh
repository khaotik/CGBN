#pragma once

typedef enum {
  cgbn_no_error=0,
  cgbn_unsupported_threads_per_instance=1,
  cgbn_unsupported_size=2,
  cgbn_unsupported_limbs_per_thread=3,
  cgbn_unsupported_operation=4,
  cgbn_threads_per_block_mismatch=5,
  cgbn_threads_per_instance_mismatch=6,
  cgbn_division_by_zero_error=7,
  cgbn_division_overflow_error=8,
  cgbn_invalid_montgomery_modulus_error=9,
  cgbn_modulus_not_odd_error=10,
  cgbn_inverse_does_not_exist_error=11,
} cgbn_error_t;
typedef struct {
  volatile cgbn_error_t _error;
  uint32_t              _instance;
  dim3                  _threadIdx;
  dim3                  _blockIdx;
} cgbn_error_report_t;
typedef enum {
  cgbn_no_checks = 0,       /* disable error checking - improves performance */
  cgbn_report_monitor = 1,  /* writes errors to the reporter object, no other actions */
  cgbn_print_monitor = 2,   /* writes errors to the reporter and prints the error to stdout */
  cgbn_halt_monitor = 15,    /* writes errors to the reporter and halts */
} cgbn_monitor_t;
template<uint32_t bits>
struct cgbn_mem_t {
  uint32_t _limbs[(bits+31)/32];
};

