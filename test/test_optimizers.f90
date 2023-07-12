program test_optimizers
  use nf, only: dense, input, network
  use iso_fortran_env, only: stderr => error_unit
  use nf_optimizers, only: sgd, rmsprop
  implicit none
  type(network) :: net_sgd, net_momentum, net_nesterov, net_rmsprop
  real, allocatable :: x(:), y(:)
  integer, parameter :: num_iterations = 1000
  real :: tolerance = 1e-3
  integer :: n
  logical :: ok_sgd = .true., ok_momentum = .true., ok_nesterov = .true., ok_rmsprop = .true.

  print '("Testing Optimizers")'
  print '(25("-"))'

  ! Network with SGD optimizer
  net_sgd = network([ &
    input(3), &
    dense(5), &
    dense(2) &
  ])

  print '("SGD Optimizer")'
  print '(20("-"))'

  x = [0.2, 0.4, 0.6]
  y = [0.123456, 0.246802]

  do n = 0, num_iterations

    call net_sgd % forward(x)
    call net_sgd % backward(y)
    call net_sgd % update(optimizer=sgd(learning_rate=1.))

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
      'sgd should converge in simple training.. failed'
      ok_sgd = .false.
    end if

  end do

  ! Network with SGD optimizer and classic momentum
  net_momentum = network([ &
    input(3), &
    dense(5), &
    dense(2) &
  ])

  print '("SGD Optimizer with Classic Momentum")'
  print '(40("-"))'

  do n = 0, num_iterations

    call net_momentum % forward(x)
    call net_momentum % backward(y)
    call net_momentum % update(optimizer=sgd(learning_rate=1., momentum=0.9))

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
      'sgd with classic momentum should converge in simple training.. failed'
      ok_momentum = .false.
    end if

  end do

  ! Network with SGD optimizer and momentum with Nesterov
  net_nesterov = network([ &
    input(3), &
    dense(5), &
    dense(2) &
  ])

  print '("SGD Optimizer with Momentum + Nesterov")'
  print '(43("-"))'

  do n = 0, num_iterations

    call net_nesterov % forward(x)
    call net_nesterov % backward(y)
    call net_nesterov % update(optimizer=sgd(learning_rate=1., momentum=0.9, nesterov=.true.))

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
      'sgd with momentum + Nesterov should converge in simple training.. failed'
      ok_nesterov = .false.
    end if

  end do

  ! Network with RMSProp optimizer
  net_rmsprop = network([ &
    input(3), &
    dense(5), &
    dense(2) &
  ])

  print '("RMSProp Optimizer")'
  print '(25("-"))'

  do n = 0, num_iterations

    call net_rmsprop % forward(x)
    call net_rmsprop % backward(y)
    call net_rmsprop % update(optimizer=rmsprop(learning_rate=0.01, decay_rate=0.9))

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
      'RMSProp optimizer should converge in simple training.. failed'
      ok_rmsprop = .false.
    end if

  end do

  if (ok_sgd .and. ok_momentum .and. ok_nesterov .and. ok_rmsprop) then
    print '(a)', 'test_optimizers: All tests passed.'
  else
    write(stderr, '(a)') 'test_optimizers: One or more tests failed.'
    stop 1
  end if
end program test_optimizers
