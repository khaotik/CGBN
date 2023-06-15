#pragma once
#include "cgbn/cgbn_types.cuh"

cudaError_t cgbn_error_report_alloc(cgbn_error_report_t **report);
cudaError_t cgbn_error_report_free(cgbn_error_report_t *report);
bool        cgbn_error_report_check(cgbn_error_report_t *report);
void        cgbn_error_report_reset(cgbn_error_report_t *report);
const char *cgbn_error_string(cgbn_error_report_t *report);
