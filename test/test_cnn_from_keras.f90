program test_cnn_from_keras

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: network
  use nf_datasets, only: download_and_unpack, keras_cnn_mnist_url

  implicit none

  type(network) :: net
  character(*), parameter :: test_data_path = 'keras_cnn_mnist.h5'
  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_cnn_mnist_url)

  net = network(test_data_path)

end program test_cnn_from_keras
