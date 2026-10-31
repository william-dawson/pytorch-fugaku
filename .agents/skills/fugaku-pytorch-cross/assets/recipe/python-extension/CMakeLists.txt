cmake_minimum_required(VERSION 3.27)
project(fugaku_python_extension C)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_library(torch_C MODULE "${PYCROSS_ROOT}/src/pytorch/torch/csrc/stub.c")
target_link_libraries(torch_C PRIVATE Python::Module "${PYCROSS_ROOT}/python-prefix/lib/libtorch_python.so")
target_link_options(torch_C PRIVATE "LINKER:-rpath-link,${FUGAKU_FJ}/lib64" "LINKER:-rpath-link,${PYCROSS_ROOT}/target/lib")
set_target_properties(torch_C PROPERTIES PREFIX "" OUTPUT_NAME "_C"
  SUFFIX ".cpython-311-aarch64-linux-gnu.so" BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "$ORIGIN/lib" LIBRARY_OUTPUT_DIRECTORY "${PYCROSS_ROOT}/python-extension-output")
