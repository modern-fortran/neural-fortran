program test_optimizers
  use nf, only: dense, input, network
  use nf_loss, only: quadratic
  use iso_fortran_env, only: stderr => error_unit
  use nf_optimizers, only: sgd, rmsprop
  implicit none
  type(network) :: net(5)
  real, allocatable :: x(:), y(:)
  real, allocatable :: ypred(:)
  integer, parameter :: num_iterations = 1000
  real :: tolerance = 1e-3
  integer :: n, i
  logical :: ok_sgd = .true., ok_momentum = .true., ok_nesterov = .true., ok_rmsprop = .true.
  logical :: converged = .false.
  ! Initialize loss and prev_loss
  real :: loss, prev_loss
  prev_loss = 0.0

  ! Instantiate a network and copy an instance to the rest of the array
  net(1) = network([input(3), dense(5), dense(2)])
  net(2:) = net(1)

  x = [0.2, 0.4, 0.6]
  y = [0.123456, 0.246802]

  do n = 0, num_iterations

    call net(1) % forward(x)
    call net(1) % backward(y)
    call net(1) % update(optimizer=sgd(learning_rate=1.))

    if (mod(n, 10) == 0) then
      ypred = net(1) % predict(x)
      converged = check_convergence(ypred, y, prev_loss)
      if (converged) then
        exit
      end if
      prev_loss = loss
    end if

  end do

  if (.not. converged) then
    write(stderr, '(a)') &
      'sgd should converge in simple training.. failed'
    ok_sgd = .false.
  end if

  ! Resetting loss and convergence flag
  converged = .false.
  prev_loss = 0.0

  do n = 0, num_iterations

    call net(2) % forward(x)
    call net(2) % backward(y)
    call net(2) % update(optimizer=sgd(learning_rate=1., momentum=0.9))

    if (mod(n, 10) == 0) then
      ypred = net(2) % predict(x)
      converged = check_convergence(ypred, y, prev_loss)
      if (converged) then
        exit
      end if
      prev_loss = loss
    end if

  end do

  if (.not. converged) then
    write(stderr, '(a)') &
      'sgd(momentum) should converge in simple training.. failed'
    ok_momentum = .false.
  end if

  ! Resetting loss and convergence flag
  converged = .false.
  prev_loss = 0.0

  do n = 0, num_iterations

    call net(3) % forward(x)
    call net(3) % backward(y)
    call net(3) % update(optimizer=sgd(learning_rate=1., momentum=0.9, nesterov=.true.))

    if (mod(n, 10) == 0) then
      ypred = net(3) % predict(x)
      converged = check_convergence(ypred, y, prev_loss)
      if (converged) then
        exit
      end if
      prev_loss = loss
    end if

  end do

  if (.not. converged) then
    write(stderr, '(a)') &
      'sgd(nesterov) should converge in simple training.. failed'
    ok_nesterov = .false.
  end if

  ! Resetting loss and convergence flag
  converged = .false.
  prev_loss = 0.0

  do n = 0, num_iterations

    call net(4) % forward(x)
    call net(4) % backward(y)
    call net(4) % update(optimizer=rmsprop(learning_rate=0.01, decay_rate=0.9))

    if (mod(n, 10) == 0) then
      ypred = net(4) % predict(x)
      converged = check_convergence(ypred, y, prev_loss)
      if (converged) then
        exit
      end if
      prev_loss = loss
    end if

  end do

  if (.not. converged) then
    write(stderr, '(a)') &
    'rmsprop should converge in simple training.. failed'
    ok_rmsprop = .false.
  end if

  if (ok_sgd .and. ok_momentum .and. ok_nesterov .and. ok_rmsprop) then
    print '(a)', 'test_optimizers: All tests passed.'
  else
    write(stderr, '(a)') 'test_optimizers: One or more tests failed.'
    stop 1
  end if

  contains

 logical function check_convergence(ypred, y, prev_loss) result(converged)
    real, intent(in) :: ypred(:), y(:), prev_loss

    loss = sqrt(sum((ypred - y)**2) / size(y))
    ! Check convergence.
    if (abs(loss - prev_loss) < tolerance) then
      converged = .true.
    end if

  end function check_convergence

end program test_optimizers
