program test_io_hdf5

  use iso_fortran_env, only: stderr => error_unit
  use nf_datasets, only: download_and_unpack, keras_model_dense_mnist_url
  use nf_io_hdf5, only: get_hdf5_attribute_string

  implicit none

  character(:), allocatable :: attr
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'
  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_model_dense_mnist_url)

  attr = get_hdf5_attribute_string(test_data_path, '.', 'backend')

  if (.not. attr == 'tensorflow') then
    ok = .false.
    write(stderr, '(a)') &
      'HDF5 variable length string attribute not read correctly.. failed'
  end if

  if (ok) then
    print '(a)', 'test_io_hdf5: All tests passed.'
  else
    write(stderr, '(a)') 'test_io_hdf5: One or more tests failed.'
    stop 1
  end if

end program test_io_hdf5
