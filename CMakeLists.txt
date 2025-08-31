# CMake version, project name, language
cmake_minimum_required(VERSION 3.20)

# If build type not specified, default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build Release")
endif()

project(neural-fortran LANGUAGES Fortran)

enable_testing()

include(cmake/options.cmake)
include(cmake/compilers.cmake)

# library to archive (libneural-fortran.a)
add_library(neural-fortran
  src/nf.f90
  src/nf/nf_activation.f90
  src/nf/nf_base_layer.f90
  src/nf/nf_conv1d_layer.f90
  src/nf/nf_conv1d_layer_submodule.f90
  src/nf/nf_conv2d_layer.f90
  src/nf/nf_conv2d_layer_submodule.f90
  src/nf/nf_cross_attention_layer.f90
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
  src/nf/nf_input2d_layer.f90
  src/nf/nf_input2d_layer_submodule.f90
  src/nf/nf_input3d_layer.f90
  src/nf/nf_input3d_layer_submodule.f90
  src/nf/nf_layer_constructors.f90
  src/nf/nf_layer_constructors_submodule.f90
  src/nf/nf_layernorm.f90
  src/nf/nf_layernorm_submodule.f90
  src/nf/nf_layer.f90
  src/nf/nf_layer_submodule.f90
  src/nf/nf_locally_connected2d_layer_submodule.f90
  src/nf/nf_locally_connected2d_layer.f90
  src/nf/nf_linear2d_layer.f90
  src/nf/nf_linear2d_layer_submodule.f90
  src/nf/nf_embedding_layer.f90
  src/nf/nf_embedding_layer_submodule.f90
  src/nf/nf_loss.f90
  src/nf/nf_loss_submodule.f90
  src/nf/nf_maxpool1d_layer.f90
  src/nf/nf_maxpool1d_layer_submodule.f90
  src/nf/nf_maxpool2d_layer.f90
  src/nf/nf_maxpool2d_layer_submodule.f90
  src/nf/nf_metrics.f90
  src/nf/nf_multihead_attention_layer.f90
  src/nf/nf_multihead_attention_layer_submodule.f90
  src/nf/nf_network.f90
  src/nf/nf_network_submodule.f90
  src/nf/nf_optimizers.f90
  src/nf/nf_parallel.f90
  src/nf/nf_parallel_submodule.f90
  src/nf/nf_random.f90
  src/nf/nf_reshape2d_layer.f90
  src/nf/nf_reshape2d_layer_submodule.f90
  src/nf/nf_reshape3d_layer.f90
  src/nf/nf_reshape3d_layer_submodule.f90
  src/nf/nf_self_attention_layer.f90
  src/nf/io/nf_io_binary.f90
  src/nf/io/nf_io_binary_submodule.f90
  src/nf/nf_dropout_layer.f90
  src/nf/nf_dropout_layer_submodule.f90
)

target_link_libraries(neural-fortran PRIVATE)
set_target_properties(neural-fortran PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

install(TARGETS neural-fortran)

# Install the module files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/ DESTINATION include)

# Also install the cmake/Findneural-fortran.cmake file
install(FILES cmake/Findneural-fortran.cmake DESTINATION include/cmake/neural-fortran)

# Remove leading or trailing whitespace
string(REGEX REPLACE "^ | $" "" LIBS "${LIBS}")

if(${PROJECT_NAME}_BUILD_TESTING)
  message(STATUS "Building tests")
  add_subdirectory(test)
endif()

if(${PROJECT_NAME}_BUILD_EXAMPLES)
  message(STATUS "Building examples")
  add_subdirectory(example)
endif()
