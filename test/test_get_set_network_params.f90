program test_get_set_network_params
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv2d, dense, flatten, input, maxpool2d, network
  implicit none
  type(network) :: net
  integer :: n
  logical :: ok = .true.
  real :: test_params_dense(8) = [1, 2, 3, 4, 5, 6, 7, 8]
  real :: test_params_conv2d(10) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

  ! First test get_num_params()
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

  ! Next, test set_params() and get_params() for a dense layer
  net = network([ &
    input(3), &
    dense(2) &
  ])

  call net % set_params(test_params_dense)

  if (.not. all(net % get_params() == test_params_dense)) then
    ok = .false.
    write(stderr, '(a)') 'Dense network params match the params that we set to it.. failed'
  end if

  if (.not. all(net % get_params() == net % layers(2) % get_params())) then
    ok = .false.
    write(stderr, '(a)') 'Single dense layer network params match that layer''s params.. failed'
  end if

  ! Finally, test set_params() and get_params() for a conv2d layer
  net = network([ &
    input([1, 3, 3]), &
    conv2d(filters=1, kernel_size=3) &
  ])

  call net % set_params(test_params_conv2d)

  if (.not. all(net % get_params() == test_params_conv2d)) then
    ok = .false.
    write(stderr, '(a)') 'Conv network params match the params that we set to it.. failed'
  end if

  if (.not. all(net % get_params() == net % layers(2) % get_params())) then
    ok = .false.
    write(stderr, '(a)') 'Single conv2d layer network params match that layer''s params.. failed'
  end if

  if (ok) then
    print '(a)', 'test_get_set_network_params: All tests passed.'
  else
    write(stderr, '(a)') 'test_get_set_network_params: One or more tests failed.'
    stop 1
  end if

end program test_get_set_network_params
