cmake_minimum_required(VERSION 3.27)
project(fjmpi_port CXX)
find_package(Torch REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_library(fjmpi_port MODULE c10d_mpi.cpp)
target_compile_features(fjmpi_port PRIVATE cxx_std_17)
target_compile_definitions(fjmpi_port PRIVATE OMPI_SKIP_MPICXX=1 _GLIBCXX_USE_CXX11_ABI=1)
target_include_directories(fjmpi_port PRIVATE ${TORCH_INCLUDE_DIRS}
  "${PYCROSS_ROOT}/src/pytorch" "${PYCROSS_ROOT}/src/pytorch/third_party/pybind11/include")
target_link_libraries(fjmpi_port PRIVATE ${TORCH_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX Python::Module
  "${PYCROSS_ROOT}/wheel-install/torch/lib/libtorch_python.so")
target_link_options(fjmpi_port PRIVATE
  "LINKER:-rpath-link,${FUGAKU_FJ}/lib64" "LINKER:-rpath-link,${PYCROSS_ROOT}/wheel-install/torch/lib")
# CPython extension APIs use the standard Linux module loader resolution.
set_target_properties(fjmpi_port PROPERTIES PREFIX "" OUTPUT_NAME "_c10d"
  NO_SYSTEM_FROM_IMPORTED ON
  SUFFIX ".cpython-311-aarch64-linux-gnu.so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fjmpi_port"
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${PYCROSS_ROOT}/wheel-install/torch/lib;${FUGAKU_FJ}/lib64")
