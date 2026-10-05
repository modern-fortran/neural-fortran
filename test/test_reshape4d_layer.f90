program test_reshape4d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: input, network, reshape4d => reshape

  implicit none

  type(network) :: net
  real, allocatable :: sample_input(:), output(:,:,:,:)
  integer, parameter :: output_shape(4) = [1, 4, 4, 4]
  integer, parameter :: input_size = product(output_shape)
  logical :: ok = .true.

  ! Create the network
  net = network([ &
    input(input_size), &
    reshape4d(output_shape(1), output_shape(2), output_shape(3), output_shape(4)) &
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

  if (ok) then
    print '(a)', 'test_reshape4d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_reshape4d_layer: One or more tests failed.'
    stop 1
  end if

end program test_reshape4d_layer
