foreach(execid
  cnn_mnist
  cnn_mnist_1d
  dense_mnist
  get_set_network_params
  network_parameters
  simple
  sine
  quadratic
  mha_simple
)
  add_executable(${execid} ${execid}.f90)
  target_link_libraries(${execid} PRIVATE
    neural-fortran
    ${LIBS}
  )
endforeach()
