CGBN Beta Release (October 2018)

This beta release has several important improvements over the alpha release:

*  Support for threads per instance (TPI) of 4, 8, and 16 in addition to the original 32
*  Support for all sizes from 32 bits to 32K bits, provided the size is evenly divisible by 32
*  Performance improvements when for sizes<2K bits, with TPI=8
*  C style wrappers for all cgbn_env_t methods


Minor update (June 2021)

*  Added build tags for turing and ampere
*  Fixed the depricated gtest TEST_CASE_P warnings

Refactor and QoL (fork repo only) (July 2023)

* CMake support
* Refactor unit tests into several compilation unit
* New methods `all_equals_ui32` `all_set_ui32`
