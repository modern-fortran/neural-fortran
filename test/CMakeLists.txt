foreach(execid
  input1d_layer
  input2d_layer
  input3d_layer
  dropout_layer
  linear2d_layer
  parametric_activation
  dense_layer
  conv1d_layer
  conv2d_layer
  locally_connected_1d_layer
  maxpool1d_layer
  maxpool2d_layer
  flatten_layer
  insert_flatten
  reshape_layer
  reshape2d_layer
  multihead_attention_layer
  dense_network
  get_set_network_params
  conv2d_network
  optimizers
  loss
  metrics
  )
  add_executable(test_${execid} test_${execid}.f90)
  target_link_libraries(test_${execid} PRIVATE neural-fortran ${LIBS})

  add_test(NAME test_${execid} COMMAND test_${execid})
endforeach()
