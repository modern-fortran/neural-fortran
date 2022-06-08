# CMake version, project name, language
cmake_minimum_required(VERSION 3.20)

# If build type not specified, default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build Release")
endif()

project(neural-fortran
LANGUAGES C Fortran
)

enable_testing()

include(FetchContent)

include(cmake/options.cmake)
include(cmake/compilers.cmake)

include(cmake/h5fortran.cmake)
include(cmake/json.cmake)

# library to archive (libneural.a)
add_library(neural
  src/nf.f90
  src/nf/nf_activation.f90
  src/nf/nf_base_layer.f90
  src/nf/nf_base_layer_submodule.f90
  src/nf/nf_conv2d_layer.f90
  src/nf/nf_conv2d_layer_submodule.f90
  src/nf/nf_datasets.f90
  src/nf/nf_datasets_submodule.f90
  src/nf/nf_datasets_mnist.f90
  src/nf/nf_datasets_mnist_submodule.f90
  src/nf/nf_dense_layer.f90
  src/nf/nf_dense_layer_submodule.f90
  src/nf/nf_flatten_layer.f90
  src/nf/nf_flatten_layer_submodule.f90
  src/nf/nf_input1d_layer.f90
  src/nf/nf_input1d_layer_submodule.f90
  src/nf/nf_input3d_layer.f90
  src/nf/nf_input3d_layer_submodule.f90
  src/nf/nf_keras.f90
  src/nf/nf_keras_submodule.f90
  src/nf/nf_layer_constructors.f90
  src/nf/nf_layer_constructors_submodule.f90
  src/nf/nf_layer.f90
  src/nf/nf_layer_submodule.f90
  src/nf/nf_loss.f90
  src/nf/nf_loss_submodule.f90
  src/nf/nf_maxpool2d_layer.f90
  src/nf/nf_maxpool2d_layer_submodule.f90
  src/nf/nf_network.f90
  src/nf/nf_network_submodule.f90
  src/nf/nf_optimizers.f90
  src/nf/nf_parallel.f90
  src/nf/nf_parallel_submodule.f90
  src/nf/nf_random.f90
  src/nf/nf_random_submodule.f90
  src/nf/io/nf_io_binary.f90
  src/nf/io/nf_io_binary_submodule.f90
  src/nf/io/nf_io_hdf5.f90
  src/nf/io/nf_io_hdf5_submodule.f90
)
target_link_libraries(neural PRIVATE h5fortran::h5fortran HDF5::HDF5 jsonfortran::jsonfortran)

install(TARGETS neural)

# Remove leading or trailing whitespace
string(REGEX REPLACE "^ | $" "" LIBS "${LIBS}")

# tests

foreach(execid input1d_layer input3d_layer dense_layer conv2d_layer maxpool2d_layer flatten_layer dense_network dense_network_from_keras conv2d_network io_hdf5 keras_read_model)
  add_executable(test_${execid} test/test_${execid}.f90)
  target_link_libraries(test_${execid} PRIVATE neural h5fortran::h5fortran jsonfortran::jsonfortran ${LIBS})
  add_test(test_${execid} bin/test_${execid})
endforeach()

foreach(execid cnn mnist mnist_from_keras simple sine)
  add_executable(${execid} example/${execid}.f90)
  target_link_libraries(${execid} PRIVATE neural h5fortran::h5fortran jsonfortran::jsonfortran ${LIBS})
endforeach()
