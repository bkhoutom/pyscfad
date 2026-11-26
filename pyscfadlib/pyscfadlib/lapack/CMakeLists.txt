nanobind_add_module(pyscfad_lapack lapack.cc lapack_kernels.cc STABLE_ABI)
set_target_properties(pyscfad_lapack PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
# Function pointers come from scipy.linalg.cython_lapack at import time.
install(TARGETS pyscfad_lapack LIBRARY DESTINATION pyscfadlib RUNTIME DESTINATION pyscfadlib)
