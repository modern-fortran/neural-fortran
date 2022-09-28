program test_reshape_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: input, network, reshape_layer => reshape
  use nf_datasets, only: download_and_unpack, keras_reshape_url

  implicit none

  type(network) :: net
  real, allocatable :: sample_input(:), output(:,:,:)
  integer, parameter :: output_shape(3) = [3, 32, 32]
  integer, parameter :: input_size = product(output_shape)
  character(*), parameter :: keras_reshape_path = 'keras_reshape.h5'
  logical :: file_exists
  logical :: ok = .true.

  ! Create the network
  net = network([ &
    input(input_size), &
    reshape_layer(output_shape) &
  ])

  if (.not. size(net % layers) == 2) then
    write(stderr, '(a)') 'the network should have 2 layers.. failed'
    ok = .false.
  end if

  ! Initialize test data
  allocate(sample_input(input_size))
  call random_number(sample_input)

  ! Propagate forward and get the output
  call net % forward(sample_input)
  call net % layers(2) % get_output(output)

  if (.not. all(shape(output) == output_shape)) then
    write(stderr, '(a)') 'the reshape layer produces expected output shape.. failed'
    ok = .false.
  end if

  if (.not. all(reshape(sample_input, output_shape) == output)) then
    write(stderr, '(a)') 'the reshape layer produces expected output values.. failed'
    ok = .false.
  end if

  ! Now test reading the reshape layer from a Keras h5 model.
  inquire(file=keras_reshape_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_reshape_url)

  net = network(keras_reshape_path)

  if (.not. size(net % layers) == 2) then
    write(stderr, '(a)') 'the reshape network from Keras has the correct size.. failed'
    ok = .false.
  end if

  if (.not. net % layers(2) % name == 'reshape') then
    write(stderr, '(a)') 'the 2nd layer of the reshape network from Keras is a reshape layer.. failed'
    ok = .false.
  end if

  ! Test that the output shape checks out
  call net % layers(1) % get_output(sample_input)
  call net % layers(2) % get_output(output)

  if (.not. all(shape(output) == [1, 28, 28])) then
    write(stderr, '(a)') 'the target shape of the reshape layer is correct.. failed'
    ok = .false.
  end if

  if (ok) then
    print '(a)', 'test_reshape_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_reshape_layer: One or more tests failed.'
    stop 1
  end if

end program test_reshape_layer
