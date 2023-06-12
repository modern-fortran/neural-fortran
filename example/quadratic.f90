program quadratic_fit
  ! This program fits a quadratic function using a small neural network using
  ! stochastic gradient descent, batch gradient descent, and minibatch gradient
  ! descent.
  use nf, only: dense, input, network
  implicit none
  type(network) :: net_sgd, net_batch_sgd, net_minibatch_sgd
  real, dimension(:), allocatable :: x, y
  integer, parameter :: num_epochs = 1000
  integer, parameter :: train_size = 1000
  integer, parameter :: test_size = 30
  real, dimension(:), allocatable :: xtest, ytest
  real, dimension(:), allocatable :: ypred_sgd, ypred_batch_sgd, ypred_minibatch_sgd
  integer :: i, n, batch_size
  real :: learning_rate

  print '("Fitting quadratic function")'
  print '(60("="))'

  allocate(xtest(test_size), ytest(test_size))
  xtest = [((i - 1) * 2 / test_size, i = 1, test_size)]
  ytest = quadratic(xtest)

  ! x and y as 1-D arrays
  allocate(x(train_size), y(train_size))

  ! Generating the dataset
  do i = 1, train_size
    call random_number(x(i))
    x(i) = x(i) * 2
  end do
  y = quadratic(x)

  ! optimizer and learning rate
  learning_rate = 0.01
  batch_size = 10

  ! Instantiate a separate network for each optimization method.
  net_sgd = network([input(1), dense(3), dense(1)])
  net_batch_sgd = network([input(1), dense(3), dense(1)])
  net_minibatch_sgd = network([input(1), dense(3), dense(1)])

  ! Print network info to stdout; this will be the same for all three networks.
  call net_sgd % print_info()

  ! SGD optimizer
  call sgd_optimizer(net_sgd, x, y, learning_rate, num_epochs)

  ! Batch SGD optimizer
  call batch_gd_optimizer(net_batch_sgd, x, y, learning_rate, num_epochs)

  ! Mini-batch SGD optimizer
  call minibatch_gd_optimizer(net_minibatch_sgd, x, y, learning_rate, num_epochs, batch_size)

  ! Calculate predictions on the test set
  ypred_sgd = [(net_sgd % predict([xtest(i)]), i = 1, test_size)]
  ypred_batch_sgd = [(net_batch_sgd % predict([xtest(i)]), i = 1, test_size)]
  ypred_minibatch_sgd = [(net_minibatch_sgd % predict([xtest(i)]), i = 1, test_size)]

  ! Print the mean squared error
  print '("Stochastic gradient descent MSE:", f9.6)', sum((ypred_sgd - ytest)**2) / size(ytest)
  print '("     Batch gradient descent MSE: ", f9.6)', sum((ypred_batch_sgd - ytest)**2) / size(ytest)
  print '(" Minibatch gradient descent MSE: ", f9.6)', sum((ypred_minibatch_sgd - ytest)**2) / size(ytest)

contains

  real elemental function quadratic(x) result(y)
    ! Quadratic function
    real, intent(in) :: x
    y = (x**2 / 2 + x / 2 + 1) / 2
  end function quadratic

  subroutine sgd_optimizer(net, x, y, learning_rate, num_epochs)
    ! In the stochastic gradient descent (SGD) optimizer, we run the forward
    ! and backward passes and update the weights for each training sample,
    ! one at a time.
    type(network), intent(inout) :: net
    real, dimension(:), intent(in) :: x, y
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_epochs
    integer :: i, n

    print *, "Running SGD optimizer..."

    do n = 1, num_epochs
      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
        call net % update(learning_rate)
      end do
    end do

  end subroutine sgd_optimizer

  subroutine batch_gd_optimizer(net, x, y, learning_rate, num_epochs)
    ! Like the stochastic gradient descent (SGD) optimizer, except that here we
    ! accumulate the weight gradients for all training samples and update the
    ! weights once per epoch.
    type(network), intent(inout) :: net
    real, dimension(:), intent(in) :: x, y
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_epochs
    integer :: i, n

    print *, "Running batch GD optimizer..."

    do n = 1, num_epochs
      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
      end do
      call net % update(learning_rate / size(x))
    end do

  end subroutine batch_gd_optimizer

  subroutine minibatch_gd_optimizer(net, x, y, learning_rate, num_epochs, batch_size)
    ! Like the batch SGD optimizer, except that here we accumulate the weight
    ! over a number of mini batches and update the weights once per mini batch.
    !
    ! Note: -O3 on GFortran must be accompanied with -fno-frontend-optimize for
    ! this subroutine to converge to a solution.
    type(network), intent(inout) :: net
    real, dimension(:), intent(in) :: x, y
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_epochs, batch_size
    integer :: i, j, n, num_samples, num_batches, start_index, end_index
    real, dimension(:), allocatable :: batch_x, batch_y
    integer, dimension(:), allocatable :: batch_indices

    print *, "Running mini-batch GD optimizer..."

    num_samples = size(x)
    num_batches = num_samples / batch_size

    ! Generate shuffled indices for the mini-batches
    allocate(batch_x(batch_size), batch_y(batch_size))
    allocate(batch_indices(num_batches))

    do j = 1, num_batches
      batch_indices(j) = (j - 1) * batch_size + 1
    end do

    call shuffle(batch_indices)

    do n = 1, num_epochs
      do j = 1, num_batches
        start_index = batch_indices(j)
        end_index = min(start_index + batch_size - 1, num_samples)

        do i = start_index, end_index
          call net % forward([x(i)])
          call net % backward([y(i)])
        end do

        call net % update(learning_rate / batch_size)
      end do
    end do
  end subroutine minibatch_gd_optimizer

  subroutine shuffle(arr)
    ! Shuffle an array using the Fisher-Yates algorithm.
    integer, dimension(:), intent(inout) :: arr
    real :: j
    integer :: i, temp

    do i = size(arr), 2, -1
      call random_number(j)
      j = floor(j * real(i)) + 1.0
      temp = arr(i)
      arr(i) = arr(int(j))
      arr(int(j)) = temp
    end do
  end subroutine shuffle

end program quadratic_fit