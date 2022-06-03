program test_io_hdf5

  use iso_fortran_env, only: int64, stderr => error_unit
  use nf_datasets, only: download_and_unpack, keras_model_dense_mnist_url
  use nf_io_hdf5, only: get_hdf5_attribute_string
  use h5fortran, only: hdf5_file

  implicit none

  character(:), allocatable :: attr
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'
  type(hdf5_file) :: f
  real, allocatable :: bias(:), weights(:,:)
  integer(int64), allocatable :: dims(:)

  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_model_dense_mnist_url)

  attr = get_hdf5_attribute_string(test_data_path, '.', 'backend')

  if (.not. attr == 'tensorflow') then
    ok = .false.
    write(stderr, '(a)') &
      'HDF5 variable length string attribute was read correctly.. failed'
  end if

  call f % open(test_data_path, 'r')

  call f % shape('/model_weights/dense/dense/bias:0', dims)
  allocate(bias(dims(1)))
  bias = 0
  call f % read('/model_weights/dense/dense/bias:0', bias)

  if (.not. all(dims == [30])) then
    ok = .false.
    write(stderr, '(a)') 'HDF5 1-d dataset dims inquiry is correct.. failed'
  end if

  call f % shape('/model_weights/dense/dense/kernel:0', dims)
  allocate(weights(dims(1), dims(2)))
  weights = 0
  call f % read('/model_weights/dense/dense/kernel:0', weights)

  if (.not. all(dims == [30, 784])) then
    ok = .false.
    print *, dims
    write(stderr, '(a)') 'HDF5 2-d dataset dims inquiry is correct.. failed'
  end if

  call f % close()

  if (ok) then
    print '(a)', 'test_io_hdf5: All tests passed.'
  else
    write(stderr, '(a)') 'test_io_hdf5: One or more tests failed.'
    stop 1
  end if

end program test_io_hdf5
