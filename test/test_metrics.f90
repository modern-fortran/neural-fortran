program test_metrics
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dense, input, network, sgd, mse
  implicit none
  type(network) :: net
  logical :: ok = .true.

  ! Minimal 2-layer network
  net = network([ &
    input(1), &
    dense(1) &
  ])

  training: block
    real :: x(1), y(1)
    real :: tolerance = 1e-3
    integer :: n
    integer, parameter :: num_iterations = 1000 
    real :: quadratic_loss, mse_metric
    real, allocatable :: metrics(:,:)

    x = [0.1234567]
    y = [0.7654321]

    do n = 1, num_iterations
      call net % forward(x)
      call net % backward(y)
      call net % update(sgd(learning_rate=1.))
      if (all(abs(net % predict(x) - y) < tolerance)) exit
    end do
    
    ! Returns only one metric, based on the default loss function (quadratic).
    metrics = net % evaluate(reshape(x, [1, 1]), reshape(y, [1, 1]))
    quadratic_loss = metrics(1,1)

    if (.not. all(shape(metrics) == [1, 1])) then
      write(stderr, '(a)') 'metrics array is the correct shape (1, 1).. failed'
      ok = .false.
    end if

    ! Returns two metrics, one from the loss function and another specified by the user.
    metrics = net % evaluate(reshape(x, [1, 1]), reshape(y, [1, 1]), metric=mse())

    if (.not. all(shape(metrics) == [1, 2])) then
      write(stderr, '(a)') 'metrics array is the correct shape (1, 2).. failed'
      ok = .false.
    end if

    mse_metric = metrics(1,2)

    if (.not. all(metrics < 1e-5)) then
      write(stderr, '(a)') 'value for all metrics is expected.. failed'
      ok = .false.
    end if

    if (.not. metrics(1,1) == quadratic_loss) then
      write(stderr, '(a)') 'first metric should be the same as that of the loss function.. failed'
      ok = .false.
    end if

  end block training

  if (ok) then
    print '(a)', 'test_metrics: All tests passed.'
  else
    write(stderr, '(a)') 'test_metrics: One or more tests failed.'
    stop 1
  end if

end program test_metrics
