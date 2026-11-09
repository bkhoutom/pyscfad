cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pyscfad_sgx LANGUAGES CXX)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  option(ENABLE_OPENMP "Parallel SGX grid batches" ON)
endif()

file(GLOB SGX_GENERATED CONFIGURE_DEPENDS generated/*.C)
file(GLOB SGX_FLOAT CONFIGURE_DEPENDS generated/fp32/*.C)
add_library(sgx_ad SHARED sgx.cpp generated/kernels.cpp generated/derivatives.cpp generated/hrr_derivatives.cpp
  generated/fp32/kernels.cpp generated/fp32/derivatives.cpp ${SGX_GENERATED} ${SGX_FLOAT})
target_compile_features(sgx_ad PRIVATE cxx_std_11)
if(ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(sgx_ad PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
# No ISA flags, fixed SIMD width, fast-math, or CPU-specific dispatch.
# The compiler vectorizes the grid loops for the user's compilation target.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  target_compile_options(sgx_ad PRIVATE -fno-math-errno)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  target_compile_options(sgx_ad PRIVATE -fp-model=precise)
  # precise restores math errno semantics; Boys arguments are nonnegative.
  # Apply this after the FP model so sqrt can use packed instructions.
  set_property(SOURCE sgx.cpp APPEND PROPERTY COMPILE_OPTIONS -fno-math-errno)
endif()
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set_target_properties(sgx_ad PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
install(TARGETS sgx_ad LIBRARY DESTINATION pyscfadlib RUNTIME DESTINATION pyscfadlib)
install(FILES codegen/upstream/LICENSE DESTINATION pyscfadlib/licenses
        RENAME snk_integral_codegen.LICENSE)
