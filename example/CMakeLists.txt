foreach(execid
  cnn
  cnn_from_keras
  dense_from_keras
  mnist
  simple
  sine
)
  add_executable(${execid} ${execid}.f90)
  target_link_libraries(${execid} PRIVATE
    neural
    h5fortran::h5fortran
    jsonfortran::jsonfortran
    ${LIBS}
  )
endforeach()
