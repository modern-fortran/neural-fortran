program test_keras_read_model

  use iso_fortran_env, only: stderr => error_unit
  use nf_datasets, only: download_and_unpack, keras_dense_mnist_url, &
    keras_cnn_mnist_url
  use nf_keras, only: get_keras_h5_layers, keras_layer

  implicit none

  character(*), parameter :: keras_dense_path = 'keras_dense_mnist.h5'
  character(*), parameter :: keras_cnn_path = 'keras_cnn_mnist.h5'

  type(keras_layer), allocatable :: keras_layers(:)

  logical :: file_exists
  logical :: ok = .true.

  ! First test the dense model

  inquire(file=keras_dense_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_dense_mnist_url)

  keras_layers = get_keras_h5_layers(keras_dense_path)

  if (size(keras_layers) /= 3) then
    ok = .false.
    write(stderr, '(a)') 'Keras dense MNIST model has 3 layers.. failed'
  end if

  if (keras_layers(1) % class /= 'InputLayer') then
    ok = .false.
    write(stderr, '(a)') 'Keras first layer should be InputLayer.. failed'
  end if

  if (.not. all(keras_layers(1) % units == [784])) then
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

  ! Now testing for the CNN model

  inquire(file=keras_cnn_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_cnn_mnist_url)

  keras_layers = get_keras_h5_layers(keras_cnn_path)

  if (.not. all(keras_layers(1) % units == [1, 28, 28])) then
    ok = .false.
    write(stderr, '(a)') &
      'Keras CNN input layer shape is expected.. failed'
  end if

  if (.not. keras_layers(2) % class == 'Conv2D') then
    ok = .false.
    write(stderr, '(a)') &
      'Keras CNN second layer is Conv2D.. failed'
  end if

  if (.not. keras_layers(2) % filters == 8) then
    ok = .false.
    write(stderr, '(a)') &
      'Keras CNN second layer number of filters is expected.. failed'
  end if

  if (.not. all(keras_layers(2) % kernel_size == [3, 3])) then
    ok = .false.
    write(stderr, '(a)') &
      'Keras CNN second layer kernel_size is expected.. failed'
  end if

  if (.not. keras_layers(3) % class == 'MaxPooling2D') then
    ok = .false.
    write(stderr, '(a)') &
      'Keras CNN third layer is MaxPooling2D.. failed'
  end if

  if (.not. all(keras_layers(3) % pool_size == [2, 2])) then
    ok = .false.
    write(stderr, '(a)') &
      'Keras CNN second layer pool_size is expected.. failed'
  end if

  if (.not. all(keras_layers(3) % strides == [2, 2])) then
    ok = .false.
    write(stderr, '(a)') &
      'Keras CNN second layer strides are expected.. failed'
  end if

  if (ok) then
    print '(a)', 'test_keras_read_model: All tests passed.'
  else
    write(stderr, '(a)') 'test_keras_read_model: One or more tests failed.'
    stop 1
  end if

end program test_keras_read_model
