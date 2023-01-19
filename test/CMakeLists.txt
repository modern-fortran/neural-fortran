foreach(execid
  input1d_layer
  input3d_layer
  dense_layer
  conv2d_layer
  maxpool2d_layer
  flatten_layer
  insert_flatten
  reshape_layer
  dense_network
  get_set_network_params
  io_hdf5
  keras_read_model
  dense_network_from_keras
  cnn_from_keras
  conv2d_network
  )
  add_executable(test_${execid} test_${execid}.f90)
  target_link_libraries(test_${execid} PRIVATE neural h5fortran::h5fortran jsonfortran::jsonfortran ${LIBS})

  add_test(NAME test_${execid} COMMAND test_${execid})
endforeach()

set_tests_properties(test_dense_network_from_keras test_io_hdf5 test_keras_read_model PROPERTIES
RESOURCE_LOCK download
)
