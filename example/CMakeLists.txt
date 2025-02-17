foreach(execid
  cnn_mnist
  dense_mnist
  get_set_network_params
  linear2d
  network_parameters
  simple
  sine
  quadratic
)
  add_executable(${execid} ${execid}.f90)
  target_link_libraries(${execid} PRIVATE
    neural-fortran
    ${LIBS}
  )
endforeach()
