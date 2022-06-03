program test_dense_network_from_keras

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: network
  use nf_datasets, only: download_and_unpack, keras_model_dense_mnist_url

  implicit none

  type(network) :: net
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'

  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_model_dense_mnist_url)

  net = network(test_data_path)

  if (.not. size(net % layers) == 3) then
    write(stderr, '(a)') 'dense network should have 3 layers.. failed'
    ok = .false.
  end if

  if (.not. net % layers(1) % name == 'input') then
    write(stderr, '(a)') 'First layer should be an input layer.. failed'
    ok = .false.
  end if

  if (.not. all(net % layers(1) % layer_shape == [784])) then
    write(stderr, '(a)') 'First layer should have shape [784].. failed'
    ok = .false.
  end if

  if (.not. net % layers(2) % name == 'dense') then
    write(stderr, '(a)') 'Second layer should be a dense layer.. failed'
    ok = .false.
  end if

  if (.not. all(net % layers(2) % layer_shape == [30])) then
    write(stderr, '(a)') 'Second layer should have shape [30].. failed'
    ok = .false.
  end if

  if (.not. net % layers(3) % name == 'dense') then
    write(stderr, '(a)') 'Third layer should be a dense layer.. failed'
    ok = .false.
  end if

  if (.not. all(net % layers(3) % layer_shape == [10])) then
    write(stderr, '(a)') 'Third layer should have shape [10].. failed'
    ok = .false.
  end if

  if (ok) then
    print '(a)', 'test_dense_network_from_keras: All tests passed.'
  else
    write(stderr, '(a)') &
      'test_dense_network_from_keras: One or more tests failed.'
    stop 1
  end if

end program test_dense_network_from_keras
