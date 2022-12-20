program test_get_set_network_params
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv2d, dense, flatten, input, maxpool2d, network
  implicit none
  type(network) :: net
  integer :: n
  logical :: ok = .true.

  net = network([ &
    input([3, 5, 5]), & ! 5 x 5 image with 3 channels
    conv2d(filters=2, kernel_size=3), & ! kernel shape [2, 3, 3, 3], output shape [2, 3, 3], 56 parameters total
    flatten(), &
    dense(4) & ! weights shape [72], biases shape [4], 76 parameters total
  ])

  if (.not. net % get_num_params() == 132) then
    ok = .false.
    write(stderr, '(a)') 'network % get_num_params() returns an expected result.. failed'
  end if

  if (.not. all(net % layers % get_num_params() == [0, 56, 0, 76])) then
    ok = .false.
    write(stderr, '(a)') 'Sizes of layer parameters are all as expected.. failed'
  end if

  if (ok) then
    print '(a)', 'test_get_set_network_params: All tests passed.'
  else
    write(stderr, '(a)') 'test_get_set_network_params: One or more tests failed.'
    stop 1
  end if

end program test_get_set_network_params
