program quadratic_fit
  use nf, only: dense, input, network
  implicit none
  type(network) :: net
  real, dimension(:), allocatable :: x, y
  integer, parameter :: num_iterations = 100000
  integer, parameter :: test_size = 30
  real, dimension(:), allocatable :: xtest, ytest, ypred
  integer :: i, n, batch_size
  real :: learning_rate

  print '("Fitting quadratic function")'
  print '(60("="))'

  net = network([ &
    input(1), &
    dense(3), &
    dense(1) &
  ])

  call net % print_info()

  allocate(xtest(test_size), ytest(test_size), ypred(test_size))
  xtest = [((i - 1) * 2 / test_size, i = 1, test_size)]
  ytest = (xtest**2 / 2 + xtest / 2 + 1) / 2

  ! x and y as 1-D arrays
  allocate(x(num_iterations), y(num_iterations))

  ! Generating the dataset
  do i = 1, num_iterations
    call random_number(x(i))
    x(i) = x(i) * 2
    y(i) = (x(i)**2 / 2 + x(i) / 2 + 1) / 2
  end do

  ! optimizer and learning rate
  learning_rate = 0.01
  batch_size = 10


  ! SGD optimizer
  call sgd_optimizer(net, x, y, learning_rate, num_iterations)

  ! Batch SGD optimizer
  call batch_sgd_optimizer(net, x, y, learning_rate, num_iterations)

  ! Mini-batch SGD optimizer
  call minibatch_sgd_optimizer(net, x, y, learning_rate, num_iterations, batch_size)

  ! Calculate predictions on the test set
  ypred = [(net % predict([xtest(i)]), i = 1, test_size)]

  ! Print the mean squared error
  print '(i0,1x,f9.6)', num_iterations, sum((ypred - ytest)**2) / size(ypred)

  contains

  subroutine sgd_optimizer(net, x, y, learning_rate, num_iterations)
    type(network), intent(inout) :: net
    real, dimension(:), intent(in) :: x, y
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_iterations
    integer :: i, n, num_samples

    num_samples = size(x)

    do n = 1, num_iterations
      do i = 1, num_samples
        call net % forward([x(i)])
        call net % backward([y(i)])
        ! SGD update
        call net % update(learning_rate)
      end do
    end do
  end subroutine sgd_optimizer


  subroutine batch_sgd_optimizer(net, x, y, learning_rate, num_iterations)
    type(network), intent(inout) :: net
    real, dimension(:), intent(in) :: x, y
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_iterations
    integer :: i

    call net % forward(x)
    call net % backward(y)
    ! Batch SGD update
    call net % update(learning_rate / batch_size)

    do i = 2, num_iterations
      call net % forward(x)
      call net % backward(y)
    end do

    ! Updating parameters
    call net % update(learning_rate / batch_size)
  end subroutine batch_sgd_optimizer


  subroutine minibatch_sgd_optimizer(net, x, y, learning_rate, num_iterations, batch_size)
    type(network), intent(inout) :: net
    real, dimension(:), intent(in) :: x, y
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_iterations, batch_size
    integer :: i, n, num_samples, num_batches, start_index, end_index
    real, dimension(:), allocatable :: batch_x, batch_y

    num_samples = size(x)
    num_batches = num_samples / batch_size

    allocate(batch_x(batch_size), batch_y(batch_size))

    do n = 1, num_iterations
      do i = 1, num_batches
        ! Selecting batch
        start_index = (i - 1) * batch_size + 1
        end_index = i * batch_size
        batch_x = x(start_index:end_index)
        batch_y = y(start_index:end_index)

        call net % forward(batch_x)
        call net % backward(batch_y)
        ! Mini-batch SGD update
        call net % update(learning_rate / batch_size)
      end do
    end do

  end subroutine minibatch_sgd_optimizer

end program quadratic_fit