program test_io_hdf5

  use iso_fortran_env, only: stderr => error_unit
  use nf_datasets, only: download_and_unpack, keras_dense_mnist_url
  use nf_io_hdf5, only: hdf5_attribute_string, get_hdf5_dataset

  implicit none

  character(:), allocatable :: attr
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'
  real, allocatable :: bias(:), weights(:,:)

  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_dense_mnist_url)

  attr = hdf5_attribute_string(test_data_path, '.', 'backend')

  if (.not. attr == 'tensorflow') then
    ok = .false.
    write(stderr, '(a)') &
      'HDF5 variable length string attribute was read correctly.. failed'
  end if

  ! Read 1-d real32 dataset
  call get_hdf5_dataset( &
    test_data_path, &
    '/model_weights/dense/dense/bias:0', &
    bias &
  )

  ! Read 2-d real32 dataset
  call get_hdf5_dataset(test_data_path, &
    '/model_weights/dense/dense/kernel:0', &
    weights &
  )

  if (.not. all(shape(bias) == [30])) then
    ok = .false.
    write(stderr, '(a)') 'HDF5 1-d dataset dims inquiry is correct.. failed'
  end if

  if (.not. all(shape(weights) == [784, 30])) then
    ok = .false.
    write(stderr, '(a)') 'HDF5 2-d dataset dims inquiry is correct.. failed'
  end if

  if (ok) then
    print '(a)', 'test_io_hdf5: All tests passed.'
  else
    write(stderr, '(a)') 'test_io_hdf5: One or more tests failed.'
    stop 1
  end if

end program test_io_hdf5
