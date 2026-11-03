cmake_minimum_required(VERSION 3.27)
project(fastdrop CXX)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(fastdrop SHARED fast_bernoulli.cpp)
target_compile_features(fastdrop PRIVATE cxx_std_17)
target_compile_options(fastdrop PRIVATE -O3)
target_compile_definitions(fastdrop PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_include_directories(fastdrop PRIVATE ${TORCH_INCLUDE_DIRS} "${PYCROSS_ROOT}/src/pytorch")
target_link_libraries(fastdrop PRIVATE ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_options(fastdrop PRIVATE
  "LINKER:-rpath-link,${FUGAKU_FJ}/lib64" "LINKER:-rpath-link,${PYCROSS_ROOT}/wheel-install/torch/lib")
set_target_properties(fastdrop PROPERTIES BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${PYCROSS_ROOT}/wheel-install/torch/lib;${FUGAKU_FJ}/lib64"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
