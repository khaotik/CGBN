cmake_minimum_required(VERSION 3.20)
project(CGBN)
include(CTest)
enable_testing()
# opts
option(CGBN_BUILD_TESTS "build unit tests" ON)
option(CGBN_BUILD_BENCHMARK "build benchmark" ON)
option(CGBN_BUILD_SAMPLES "build sample codes" OFF)
option(CGBN_USE_OPENMP "use OpenMP" ON)
set(CGBN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
# setup cuda
find_package(CUDA REQUIRED)
set(CUDA_LINK_LIBRARIES_KEYWORD "PRIVATE")
set(CUDA_ARCH_LIST "Auto" CACHE STRING "list of cuda architectures to compile on")
cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})
message("nvcc arch flags are ${ARCH_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

# openmp
if(${CGBN_USE_OPENMP})
  find_package(OpenMP)
  message("OMP C++ flags are ${OpenMP_CXX_FLAGS}")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=\"${OpenMP_CXX_FLAGS}\"")
endif()

# unit tests
if(${CGBN_BUILD_TESTS})
  set(CGBN_TEST_SRCDIR "${PROJECT_SOURCE_DIR}/unit_tests")
  file(GLOB_RECURSE TESTER_CU_SRCLIST "${CGBN_TEST_SRCDIR}/*.cu" "${CGBN_TEST_SRCDIR}/*.cuh" "${CGBN_TEST_SRCDIR}/*.h")
  file(GLOB_RECURSE TESTER_SRCLIST "${CGBN_TEST_SRCDIR}/*.cc")
  cuda_add_executable(tester ${TESTER_CU_SRCLIST})
  target_sources(tester PRIVATE ${TESTER_SRCLIST})
  target_include_directories(tester PRIVATE "${CGBN_INCLUDE_DIR}" "${CGBN_TEST_SRCDIR}")
  set_property(TARGET tester PROPERTY CXX_STANDARD 14)
  set_property(TARGET tester PROPERTY CXX_STANDARD_REQUIRED ON)

  set(__prev_build_so "${BUILD_SHARED_LIBS}")
  set(BUILD_SHARED_LIBS OFF)
    set(BUILD_GMOCK OFF)
    set(INSTALL_GMOCK OFF)
    set(GTEST_DIR "${PROJECT_SOURCE_DIR}/3rdparty/googletest")
    add_subdirectory("${GTEST_DIR}")
  set(BUILD_SHARED_LIBS ${__prev_build_so})

  # TODO GMP should be found and target_link_libraries(.., gmp)
  target_link_libraries(tester PRIVATE GTest::gtest_main gmp)
  if(${CGBN_USE_OPENMP})
    target_link_libraries(tester PRIVATE OpenMP::OpenMP_CXX)
  endif()

  include(GoogleTest)
  gtest_discover_tests(tester)
endif()

# benchmark
if(${CGBN_BUILD_BENCHMARK})
  set(CGBN_BENCH_SRCDIR "${PROJECT_SOURCE_DIR}/benchmark")
  add_executable(gmp_bench ${CGBN_BENCH_SRCDIR}/gmp_tester.cc)
  cuda_add_executable(xmp_bench ${CGBN_BENCH_SRCDIR}/xmp_tester.cu)

  target_link_libraries(gmp_bench PRIVATE gmp)
  target_include_directories(gmp_bench PRIVATE "${CGBN_INCLUDE_DIR}" "${CGBN_BENCH_SRCDIR}")

  target_link_libraries(xmp_bench PRIVATE gmp)
  target_include_directories(xmp_bench PRIVATE "${CGBN_INCLUDE_DIR}" "${CGBN_BENCH_SRCDIR}")

  if(${CGBN_USE_OPENMP})
    target_link_libraries(gmp_bench PRIVATE OpenMP::OpenMP_CXX)
  else()
    error("OpenMP is REQUIRED for benchmark. You may as well add -D CGBN_BUILD_BENCHMARK=OFF to cmake")
  endif()
endif()

if(${CGBN_BUILD_SAMPLES})
  set(CGBN_SAMPLES_DIR "${PROJECT_SOURCE_DIR}/samples")
  file(GLOB SAMPLE_DIRS "${CGBN_SAMPLES_DIR}/sample_*" LIST_DIRECTORIES TRUE)
  foreach(__subdir__ ${SAMPLE_DIRS})
    get_filename_component(_sample_target_ "${__subdir__}" NAME)
    file(GLOB_RECURSE __srclist__ "${__subdir__}/*.cu")
    cuda_add_executable(${_sample_target_} ${__srclist__})
    target_include_directories(${_sample_target_} PRIVATE "${CGBN_INCLUDE_DIR}")
    target_link_libraries(${_sample_target_} PRIVATE gmp)
  endforeach()
endif()
message("sample dirs are ${SAMPLE_DIRS}")

# samples
# TODO
