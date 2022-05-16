program test_conv2d_network

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv2d, input, network

  implicit none

  type(network) :: net
  real, allocatable :: sample_input(:,:,:), output(:,:,:)
  logical :: ok = .true.

  ! 3-layer convolutional network
  net = network([ &
    input([3, 32, 32]), &
    conv2d(filters=16, kernel_size=3), &
    conv2d(filters=32, kernel_size=3) &
  ])

  if (.not. size(net % layers) == 3) then
    write(stderr, '(a)') 'conv2d network should have 3 layers.. failed'
    ok = .false.
  end if

  allocate(sample_input(3, 32, 32))
  sample_input = 0

  call net % forward(sample_input)
  call net % layers(3) % get_output(output)

  if (.not. all(shape(output) == [32, 28, 28])) then
    write(stderr, '(a)') 'conv2d network output should have correct shape.. failed'
    ok = .false.
  end if

  if (ok) then
    print '(a)', 'test_conv2d_network: All tests passed.'
  else
    write(stderr, '(a)') 'test_conv2d_network: One or more tests failed.'
    stop 1
  end if

end program test_conv2d_network
