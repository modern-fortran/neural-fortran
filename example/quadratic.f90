program quadratic_fit
  ! This program fits a quadratic function using a small neural network using
  ! stochastic gradient descent, batch gradient descent, and minibatch gradient
  ! descent.
  use nf, only: dense, input, network
  use nf_dense_layer, only: dense_layer
  use nf_optimizers, only: sgd, rmsprop, adam, adagrad

  implicit none
  type(network) :: net(10)

  ! Training parameters
  integer, parameter :: num_epochs = 1000
  integer, parameter :: train_size = 1000
  integer, parameter :: test_size = 100
  integer, parameter :: batch_size = 100
  real, parameter :: learning_rate = 0.01
  real, parameter :: decay_rate = 0.9
  real, parameter :: beta1 = 0.85
  real, parameter :: beta2 = 0.95
  real, parameter :: epsilon = 1e-8

  ! Input and output data
  real, allocatable :: x(:), y(:) ! training data
  real, allocatable :: xtest(:), ytest(:) ! testing data

  integer :: i, n

  print '("Fitting quadratic function")'
  print '(60("="))'

  allocate(xtest(test_size), ytest(test_size))
  xtest = [(real(i - 1) * 2 / test_size, i = 1, test_size)]
  ytest = quadratic(xtest)

  ! x and y as 1-D arrays
  allocate(x(train_size), y(train_size))

  ! Generating the dataset
  do i = 1, train_size
    call random_number(x(i))
    x(i) = x(i) * 2
  end do
  y = quadratic(x)

  ! Instantiate a network and copy an instance to the rest of the array
  net(1) = network([input(1), dense(3), dense(1)])
  net(2:) = net(1)

  ! Print network info to stdout; this will be the same for all three networks.
  call net(1) % print_info()

  ! SGD, no momentum
  call sgd_optimizer(net(1), x, y, xtest, ytest, learning_rate, num_epochs)

  ! SGD, momentum
  call sgd_optimizer( &
    net(2), x, y, xtest, ytest, learning_rate, num_epochs, momentum=0.9 &
  )

  ! SGD, momentum with Nesterov
  call sgd_optimizer( &
    net(3), x, y, xtest, ytest, learning_rate, num_epochs, &
    momentum=0.9, nesterov=.true. &
  )

  ! Batch SGD optimizer
  call batch_gd_optimizer(net(4), x, y, xtest, ytest, learning_rate, num_epochs)

  ! Mini-batch SGD optimizer
  call minibatch_gd_optimizer( &
    net(5), x, y, xtest, ytest, learning_rate, num_epochs, batch_size &
  )

  ! RMSProp optimizer
  call rmsprop_optimizer( &
    net(6), x, y, xtest, ytest, learning_rate, num_epochs, decay_rate &
  )

  ! Adam optimizer
  call adam_optimizer( &
    net(7), x, y, xtest, ytest, learning_rate, num_epochs, &
    beta1, beta2, epsilon &
  )

  ! Adam optimizer with L2 regularization
  call adam_optimizer( &
    net(8), x, y, xtest, ytest, learning_rate, num_epochs, &
    beta1, beta2, epsilon, weight_decay_l2=1e-4 &
  )

  ! Adam optimizer with decoupled weight decay regularization
  call adam_optimizer( &
    net(9), x, y, xtest, ytest, learning_rate, num_epochs, &
    beta1, beta2, epsilon, weight_decay_decoupled=1e-5 &
  )

  ! Adagrad optimizer
  call adagrad_optimizer( &
    net(10), x, y, xtest, ytest, learning_rate, num_epochs, epsilon &
  )

contains

  real elemental function quadratic(x) result(y)
    ! Quadratic function
    real, intent(in) :: x
    y = (x**2 / 2 + x / 2 + 1) / 2
  end function quadratic

  subroutine sgd_optimizer( &
    net, x, y, xtest, ytest, learning_rate, num_epochs, momentum, nesterov &
  )
    ! In the stochastic gradient descent (SGD) optimizer, we run the forward
    ! and backward passes and update the weights for each training sample,
    ! one at a time.
    type(network), intent(inout) :: net
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: xtest(:), ytest(:)
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_epochs
    real, intent(in), optional :: momentum
    logical, intent(in), optional :: nesterov
    real, allocatable :: ypred(:)
    real :: momentum_value
    logical :: nesterov_value
    integer :: i, n

    print '(a)', 'Stochastic gradient descent'
    print '(34("-"))'

    ! Set default values for momentum and nesterov
    if (.not. present(momentum)) then
      momentum_value = 0.0
    else
      momentum_value = momentum
    end if

    if (.not. present(nesterov)) then
      nesterov_value = .false.
    else
      nesterov_value = nesterov
    end if

    do n = 1, num_epochs
      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
        call net % update( &
          sgd( &
            learning_rate=learning_rate, &
            momentum=momentum_value, &
            nesterov=nesterov_value &
          ) &
        )
      end do

      if (mod(n, num_epochs / 10) == 0) then
        ypred = [(net % predict([xtest(i)]), i = 1, size(xtest))]
        print '("Epoch: ", i4,"/",i4,", RMSE = ", f9.6)', &
          n, num_epochs, sum((ypred - ytest)**2) / size(ytest)
      end if

    end do

    print *, ''

  end subroutine sgd_optimizer

  subroutine batch_gd_optimizer( &
    net, x, y, xtest, ytest, learning_rate, num_epochs &
  )
    ! Like the stochastic gradient descent (SGD) optimizer, except that here we
    ! accumulate the weight gradients for all training samples and update the
    ! weights once per epoch.
    type(network), intent(inout) :: net
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: xtest(:), ytest(:)
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_epochs
    real, allocatable :: ypred(:)
    integer :: i, n

    print '(a)', 'Batch gradient descent'
    print '(34("-"))'

    do n = 1, num_epochs
      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
      end do
      call net % update(sgd(learning_rate=learning_rate / size(x)))

      if (mod(n, num_epochs / 10) == 0) then
        ypred = [(net % predict([xtest(i)]), i = 1, size(xtest))]
        print '("Epoch: ", i4,"/",i4,", RMSE = ", f9.6)', &
          n, num_epochs, sum((ypred - ytest)**2) / size(ytest)
      end if

    end do

    print *, ''

  end subroutine batch_gd_optimizer

  subroutine minibatch_gd_optimizer( &
    net, x, y, xtest, ytest, learning_rate, num_epochs, batch_size &
  )
    ! Like the batch SGD optimizer, except that here we accumulate the weight
    ! over a number of mini batches and update the weights once per mini batch.
    !
    ! Note: -O3 on GFortran must be accompanied with -fno-frontend-optimize for
    ! this subroutine to converge to a solution.
    type(network), intent(inout) :: net
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: xtest(:), ytest(:)
    real, intent(in) :: learning_rate
    integer, intent(in) :: num_epochs, batch_size
    integer :: i, j, n, num_samples, num_batches, start_index, end_index
    real, allocatable :: batch_x(:), batch_y(:)
    integer, allocatable :: batch_indices(:)
    real, allocatable :: ypred(:)

    print '(a)', 'Minibatch gradient descent'
    print '(34("-"))'

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

        call net % update(sgd(learning_rate=learning_rate / batch_size))
      end do

      if (mod(n, num_epochs / 10) == 0) then
        ypred = [(net % predict([xtest(i)]), i = 1, size(xtest))]
        print '("Epoch: ", i4,"/",i4,", RMSE = ", f9.6)', &
          n, num_epochs, sum((ypred - ytest)**2) / size(ytest)
      end if

    end do

    print *, ''

  end subroutine minibatch_gd_optimizer

  subroutine rmsprop_optimizer( &
    net, x, y, xtest, ytest, learning_rate, num_epochs, decay_rate &
  )
    ! RMSprop optimizer for updating weights using root mean square
    type(network), intent(inout) :: net
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: xtest(:), ytest(:)
    real, intent(in) :: learning_rate, decay_rate
    integer, intent(in) :: num_epochs
    integer :: i, j, n
    real, allocatable :: ypred(:)

    print '(a)', 'RMSProp optimizer'
    print '(34("-"))'

    do n = 1, num_epochs

      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
      end do

      call net % update( &
        rmsprop(learning_rate=learning_rate, decay_rate=decay_rate) &
      )

      if (mod(n, num_epochs / 10) == 0) then
        ypred = [(net % predict([xtest(i)]), i = 1, size(xtest))]
        print '("Epoch: ", i4,"/",i4,", RMSE = ", f9.6)', &
          n, num_epochs, sum((ypred - ytest)**2) / size(ytest)
      end if

    end do

    print *, ''

  end subroutine rmsprop_optimizer

  subroutine adam_optimizer( &
    net, x, y, xtest, ytest, learning_rate, num_epochs, beta1, beta2, epsilon, &
    weight_decay_l2, weight_decay_decoupled &
  )
    ! Adam optimizer
    type(network), intent(inout) :: net
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: xtest(:), ytest(:)
    real, intent(in) :: learning_rate, beta1, beta2, epsilon
    real, intent(in), optional :: weight_decay_l2
    real, intent(in), optional :: weight_decay_decoupled
    integer, intent(in) :: num_epochs
    real, allocatable :: ypred(:)
    integer :: i, n
    real :: weight_decay_l2_val
    real :: weight_decay_decoupled_val

    ! Set default values for weight_decay_l2
    if (.not. present(weight_decay_l2)) then
      weight_decay_l2_val = 0.0
    else
      weight_decay_l2_val = weight_decay_l2
    end if

    ! Set default values for weight_decay_decoupled
    if (.not. present(weight_decay_decoupled)) then
      weight_decay_decoupled_val = 0.0
    else
      weight_decay_decoupled_val = weight_decay_decoupled
    end if

    print '(a)', 'Adam optimizer'
    print '(34("-"))'

    do n = 1, num_epochs
      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
      end do

      call net % update( &
        adam( &
          learning_rate=learning_rate, &
          beta1=beta1, &
          beta2=beta2, &
          epsilon=epsilon, &
          weight_decay_l2=weight_decay_l2_val, &
          weight_decay_decoupled=weight_decay_decoupled_val &
        ) &
      )

      if (mod(n, num_epochs / 10) == 0) then
        ypred = [(net % predict([xtest(i)]), i = 1, size(xtest))]
        print '("Epoch: ", i4,"/",i4,", RMSE = ", f9.6)', &
          n, num_epochs, sum((ypred - ytest)**2) / size(ytest)
      end if

    end do

    print *, ''

  end subroutine adam_optimizer

  subroutine adagrad_optimizer( &
    net, x, y, xtest, ytest, learning_rate, num_epochs, epsilon &
  )
    ! Adagrad optimizer for updating weights using adaptive gradient algorithm
    type(network), intent(inout) :: net
    real, intent(in) :: x(:), y(:)
    real, intent(in) :: xtest(:), ytest(:)
    real, intent(in) :: learning_rate, epsilon
    integer, intent(in) :: num_epochs
    integer :: i, n
    real, allocatable :: ypred(:)

    print '(a)', 'Adagrad optimizer'
    print '(34("-"))'

    do n = 1, num_epochs

      do i = 1, size(x)
        call net % forward([x(i)])
        call net % backward([y(i)])
      end do

      call net % update( &
        adagrad(learning_rate=learning_rate, epsilon=epsilon) &
      )

      if (mod(n, num_epochs / 10) == 0) then
        ypred = [(net % predict([xtest(i)]), i = 1, size(xtest))]
        print '("Epoch: ", i4,"/",i4,", RMSE = ", f9.6)', &
          n, num_epochs, sum((ypred - ytest)**2) / size(ytest)
      end if

    end do

    print *, ''

  end subroutine adagrad_optimizer

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