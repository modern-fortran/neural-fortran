program test_reshape_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: input, network, reshape

  implicit none

  type(network) :: net
  real, allocatable :: sample_input(:), output(:,:,:)
  integer, parameter :: output_shape(3) = [3, 32, 32]
  integer, parameter :: input_size = product(output_shape)
  logical :: ok = .true.

  net = network([ &
    input(input_size), &
    reshape(output_shape) &
  ])

  if (.not. size(net % layers) == 2) then
    write(stderr, '(a)') 'the network should have 2 layers.. failed'
    ok = .false.
  end if

  if (ok) then
    print '(a)', 'test_reshape_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_reshape_layer: One or more tests failed.'
    stop 1
  end if

end program test_reshape_layer
