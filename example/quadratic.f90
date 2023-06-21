program quadratic_fit
  ! This program fits a quadratic function using a small neural network using
  ! stochastic gradient descent, batch gradient descent, and minibatch gradient
  ! descent.
  use nf, only: dense, input, network
  use nf_dense_layer, only: dense_layer

  implicit none
  type(network) :: net_sgd, net_batch_sgd, net_minibatch_sgd, net_rms_prop

  ! Training parameters
  integer, parameter :: num_epochs = 1000
  integer, parameter :: train_size = 1000
  integer, parameter :: test_size = 30
  integer, parameter :: batch_size = 10
  real, parameter :: learning_rate = 0.01
  real, parameter :: decay_rate = 0.9

  ! Input and output data
  real, allocatable :: x(:), y(:) ! training data
  real, allocatable :: xtest(:), ytest(:) ! testing data
  real, allocatable :: ypred_sgd(:), ypred_batch_sgd(:), ypred_minibatch_sgd(:), ypred_rms_prop(:)

  integer :: i, n

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

  ! Instantiate a separate network for each optimization method.
  net_sgd = network([input(1), dense(3), dense(1)])
  net_batch_sgd = network([input(1), dense(3), dense(1)])
  net_minibatch_sgd = network([input(1), dense(3), dense(1)])
  net_rms_prop = network([input(1), dense(3), dense(1)])

  ! Print network info to stdout; this will be the same for all three networks.
  call net_sgd % print_info()

  ! SGD optimizer
  call sgd_optimizer(net_sgd, x, y, learning_rate, num_epochs)

  ! Batch SGD optimizer
  call batch_gd_optimizer(net_batch_sgd, x, y, learning_rate, num_epochs)

  ! Mini-batch SGD optimizer
  call minibatch_gd_optimizer(net_minibatch_sgd, x, y, learning_rate, num_epochs, batch_size)

  ! RMSProp optimizer
  call rmsprop_optimizer(net_rms_prop, x, y, learning_rate, num_epochs, decay_rate)

  ! Calculate predictions on the test set
  ypred_sgd = [(net_sgd % predict([xtest(i)]), i = 1, test_size)]
  ypred_batch_sgd = [(net_batch_sgd % predict([xtest(i)]), i = 1, test_size)]
  ypred_minibatch_sgd = [(net_minibatch_sgd % predict([xtest(i)]), i = 1, test_size)]
  ypred_rms_prop = [(net_rms_prop % predict([xtest(i)]), i = 1, test_size)]

  ! Print the mean squared error
  print '("Stochastic gradient descent MSE:", f9.6)', sum((ypred_sgd - ytest)**2) / size(ytest)
  print '("     Batch gradient descent MSE: ", f9.6)', sum((ypred_batch_sgd - ytest)**2) / size(ytest)
  print '(" Minibatch gradient descent MSE: ", f9.6)', sum((ypred_minibatch_sgd - ytest)**2) / size(ytest)
  print '("                    RMSProp MSE: ", f9.6)', sum((ypred_rms_prop - ytest)**2) / size(ytest)

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
    real, intent(in) :: x(:), y(:)
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
    real, intent(in) :: x(:), y(:)
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
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_epochs, batch_size
    integer :: i, j, n, num_samples, num_batches, start_index, end_index
    real, allocatable :: batch_x(:), batch_y(:)
    integer, allocatable :: batch_indices(:)

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

  subroutine rmsprop_optimizer(net, x, y, learning_rate, num_epochs, decay_rate)
    ! RMSprop optimizer for updating weights using root mean square
    type(network), intent(inout) :: net
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: learning_rate, decay_rate
    integer, intent(in) :: num_epochs
    integer :: i, j, n
    real, parameter :: epsilon = 1e-8 ! Small constant to avoid division by zero

    ! Define a dedicated type to store the RMSprop gradients.
    ! This is needed because array sizes vary between layers and we need to
    ! keep track of gradients for each layer over time.
    ! For now this works only for dense layers.
    ! We will need to define a similar type for conv2d layers.
    type :: rms_gradient_dense
      real, allocatable :: dw(:,:)
      real, allocatable :: db(:)
    end type rms_gradient_dense

    type(rms_gradient_dense), allocatable :: rms(:)

    print *, "Running RMSprop optimizer..."

    ! Here we allocate the array or RMS gradient derived types.
    ! We need one for each dense layer, however we will allocate it to the
    ! length of all layers as it will make housekeeping easier.
    allocate(rms(size(net % layers)))

    do n = 1, num_epochs

      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
      end do

      ! RMSprop update rule
      do j = 1, size(net % layers)
        select type (this_layer => net % layers(j) % p)
          type is (dense_layer)

            ! If this is our first time here for this layer, allocate the
            ! internal RMS gradient arrays and initialize them to zero.
            if (.not. allocated(rms(j) % dw)) then
              allocate(rms(j) % dw, mold=this_layer % dw)
              allocate(rms(j) % db, mold=this_layer % db)
              rms(j) % dw = 0
              rms(j) % db = 0
            end if

            ! Update the RMS gradients using the RMSprop moving average rule
            rms(j) % dw = decay_rate * rms(j) % dw + (1 - decay_rate) * this_layer % dw**2
            rms(j) % db = decay_rate * rms(j) % db + (1 - decay_rate) * this_layer % db**2

            ! Update weights and biases using the RMSprop update rule
            this_layer % weights = this_layer % weights - learning_rate &
              / sqrt(rms(j) % dw + epsilon) * this_layer % dw
            this_layer % biases = this_layer % biases - learning_rate &
              / sqrt(rms(j) % db + epsilon) * this_layer % db

            ! We have updated the weights and biases, so we need to reset the
            ! gradients to zero for the next epoch.
            this_layer % dw = 0
            this_layer % db = 0

        end select
      end do

    end do

  end subroutine rmsprop_optimizer

  subroutine shuffle(arr)
    ! Shuffle an array using the Fisher-Yates algorithm.
    integer, intent(inout) :: arr(:)
    real :: a
    integer :: i, temp

    do i = size(arr), 2, -1
      call random_number(a)
      a = floor(a * real(i)) + 1
      temp = arr(i)
      arr(i) = arr(int(a))
      arr(int(a)) = temp
    end do
  end subroutine shuffle

end program quadratic_fit