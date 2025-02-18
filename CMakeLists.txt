cmake_minimum_required(VERSION 3.20)
set(THIS CGBN)
project(CGBN)
include(CTest)
enable_testing()
# check if is primary project (not from add_subdirectory(...))
message("source dir is ${CMAKE_SOURCE_DIR}")
message("current source dir is ${CMAKE_CURRENT_SOURCE_DIR}")
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  set(CGBN_IS_ROOT_PROJECT ON)
else()
  set(CGBN_IS_ROOT_PROJECT OFF)
endif()
# opts
option(CGBN_BUILD_TESTS "build unit tests" ${CGBN_IS_ROOT_PROJECT})
option(CGBN_BUILD_BENCHMARK "build benchmark" ${CGBN_IS_ROOT_PROJECT})
option(CGBN_BUILD_SAMPLES "build sample codes" ${CGBN_IS_ROOT_PROJECT})
option(CGBN_USE_OPENMP "use OpenMP" ON)
set(CGBN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
file(GLOB_RECURSE CGBN_HEADERS "${CGBN_INCLUDE_DIR}/*.h" "${CGBN_INCLUDE_DIR}/*.cu")
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

# (install)
add_library(${THIS} INTERFACE)
target_include_directories(${THIS} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(${THIS} INTERFACE cxx_std_14)
if(NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
  # TODO is this really correct ?
  set(CMAKE_INSTALL_INCLUDEDIR "include")
endif()
install(DIRECTORY "${CGBN_INCLUDE_DIR}/cgbn" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# unit tests
if(${CGBN_BUILD_TESTS})
  set(CGBN_TEST_SRCDIR "${PROJECT_SOURCE_DIR}/unit_tests")
  file(GLOB_RECURSE TESTER_CU_SRCLIST "${CGBN_TEST_SRCDIR}/*.cu" "${CGBN_TEST_SRCDIR}/*.cuh" "${CGBN_TEST_SRCDIR}/*.h")
  file(GLOB_RECURSE TESTER_SRCLIST "${CGBN_TEST_SRCDIR}/*.cc")
  cuda_add_executable(tester ${TESTER_CU_SRCLIST})
  target_sources(tester PRIVATE ${TESTER_SRCLIST})
  target_include_directories(tester PRIVATE "${CGBN_TEST_SRCDIR}")
  target_link_libraries(tester PRIVATE CGBN)
  set_target_properties(tester PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)

  set(_prev_build_so "${BUILD_SHARED_LIBS}")
  set(BUILD_SHARED_LIBS OFF)
    set(BUILD_GMOCK OFF)
    set(INSTALL_GMOCK OFF)
    set(GTEST_DIR "${PROJECT_SOURCE_DIR}/3rdparty/googletest")
    add_subdirectory("${GTEST_DIR}")
  set(BUILD_SHARED_LIBS ${_prev_build_so})

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
  cuda_add_executable(xmp_bench ${CGBN_BENCH_SRCDIR}/xmp_tester.cu)
  add_executable(gmp_bench ${CGBN_BENCH_SRCDIR}/gmp_tester.cc)

  target_link_libraries(gmp_bench PRIVATE gmp CGBN)
  target_include_directories(gmp_bench PRIVATE "${CGBN_BENCH_SRCDIR}")

  target_link_libraries(xmp_bench PRIVATE gmp CGBN)
  target_include_directories(xmp_bench PRIVATE "${CGBN_BENCH_SRCDIR}")

  if(${CGBN_USE_OPENMP})
    target_link_libraries(gmp_bench PRIVATE OpenMP::OpenMP_CXX)
  else()
    error("OpenMP is REQUIRED for benchmark. You may as well add -D CGBN_BUILD_BENCHMARK=OFF to cmake")
  endif()
endif()

# samples
if(${CGBN_BUILD_SAMPLES})
  set(CGBN_SAMPLES_DIR "${PROJECT_SOURCE_DIR}/samples")
  file(GLOB SAMPLE_DIRS "${CGBN_SAMPLES_DIR}/sample_*" LIST_DIRECTORIES TRUE)
  foreach(_subdir_ ${SAMPLE_DIRS})
    get_filename_component(_sample_target_ "${_subdir_}" NAME)
    file(GLOB_RECURSE _srclist_ "${_subdir_}/*.cu")
    cuda_add_executable(${_sample_target_} ${_srclist_})
    target_include_directories(${_sample_target_} PRIVATE "${CGBN_INCLUDE_DIR}")
    target_link_libraries(${_sample_target_} PRIVATE CGBN gmp)
  endforeach()
endif()
