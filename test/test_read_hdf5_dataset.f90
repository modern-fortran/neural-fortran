program test_read_hdf5_dataset

  use iso_fortran_env, only: int64
  use h5fortran, only: hdf5_file

  implicit none
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'
  type(hdf5_file) :: f
  real, allocatable :: bias(:), weights(:,:)
  integer(int64), allocatable :: dims(:)

  call f % open(test_data_path, 'r')

  call f % shape('/model_weights/dense/dense/bias:0', dims)
  allocate(bias(dims(1)))

  call f % shape('/model_weights/dense/dense/kernel:0', dims)
  allocate(weights(dims(1), dims(2)))

  call f % read('/model_weights/dense/dense/bias:0', bias)
  call f % read('/model_weights/dense/dense/kernel:0', weights)
  call f % close()

  print *, 'bias shape: ', shape(bias)
  print *, 'weights shape: ', shape(weights)
  print *, bias
  print *, weights

end program test_read_hdf5_dataset
