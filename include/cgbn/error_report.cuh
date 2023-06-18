#pragma once
#include "cgbn/cgbn_types.cuh"

namespace cgbn {
cudaError_t cgbn_error_report_alloc(ErrorReport **report);
cudaError_t cgbn_error_report_free(ErrorReport *report);
bool        cgbn_error_report_check(ErrorReport *report);
void        cgbn_error_report_reset(ErrorReport *report);
const char *cgbn_error_string(ErrorReport *report);
}
