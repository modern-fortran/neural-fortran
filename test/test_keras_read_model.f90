program test_keras_read_model

  use iso_fortran_env, only: stderr => error_unit
  use nf_datasets, only: download_and_unpack, keras_model_dense_mnist_url
  use nf_keras, only: get_keras_h5_layers, keras_layer
  use nf, only: layer, network, dense, input

  implicit none

  character(:), allocatable :: model_config_string
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'

  type(keras_layer), allocatable :: keras_layers(:)

  type(layer), allocatable :: layers(:)
  type(network) :: net

  integer :: n
  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_model_dense_mnist_url)

  keras_layers = get_keras_h5_layers(test_data_path)

  if (size(keras_layers) /= 3) then
    ok = .false.
    write(stderr, '(a)') 'Keras dense MNIST model has 3 layers.. failed'
  end if

  if (keras_layers(1) % class /= 'InputLayer') then
    ok = .false.
    write(stderr, '(a)') 'Keras first layer should be InputLayer.. failed'
  end if

  if (.not. all(keras_layers(1) % num_elements == [784])) then
    ok = .false.
    write(stderr, '(a)') 'Keras first layer should have 784 elements.. failed'
  end if

  if (allocated(keras_layers(1) % activation)) then
    ok = .false.
    write(stderr, '(a)') &
      'Keras first layer activation should not be allocated.. failed'
  end if

  if (.not. keras_layers(2) % class == 'Dense') then
    ok = .false.
    write(stderr, '(a)') &
      'Keras second and third layers should be dense.. failed'
  end if

  if (ok) then
    print '(a)', 'test_keras_read_model: All tests passed.'
  else
    write(stderr, '(a)') 'test_keras_read_model: One or more tests failed.'
    stop 1
  end if

end program test_keras_read_model
