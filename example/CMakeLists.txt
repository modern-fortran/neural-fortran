foreach(execid cnn mnist mnist_from_keras simple sine)
  add_executable(${execid} ${execid}.f90)
  target_link_libraries(${execid} PRIVATE neural h5fortran::h5fortran jsonfortran::jsonfortran ${LIBS})
endforeach()
