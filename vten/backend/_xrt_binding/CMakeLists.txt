cmake_minimum_required(VERSION 3.14)
project(vten_xrt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# XRT paths
set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT installation root")
include_directories(${XRT_ROOT}/include)
link_directories(${XRT_ROOT}/lib)

# pybind11 — auto-detect from Python environment
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE _pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_result
)
if(_pybind11_result EQUAL 0 AND _pybind11_cmake_dir)
  set(pybind11_DIR "${_pybind11_cmake_dir}")
endif()
find_package(pybind11 REQUIRED)

# Module
pybind11_add_module(vten_xrt MODULE vten_xrt.cpp)
target_link_libraries(vten_xrt PRIVATE xrt_coreutil)
