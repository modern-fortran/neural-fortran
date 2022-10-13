program cnn_mnist

  use nf, only: network, sgd, &
    input, conv2d, maxpool2d, flatten, dense, reshape, &
    load_mnist, label_digits

  implicit none

  type(network) :: net

  real, allocatable :: training_images(:,:), training_labels(:)
  real, allocatable :: validation_images(:,:), validation_labels(:)
  real, allocatable :: testing_images(:,:), testing_labels(:)
  real, allocatable :: input_reshaped(:,:,:,:)
  real :: acc
  logical :: ok
  integer :: n
  integer, parameter :: num_epochs = 10

  call load_mnist(training_images, training_labels, &
                  validation_images, validation_labels, &
                  testing_images, testing_labels)

  net = network([ &
    input(784), &
    reshape([1,28,28]), &
    conv2d(filters=8, kernel_size=3, activation='relu'), &
    maxpool2d(pool_size=2), &
    conv2d(filters=16, kernel_size=3, activation='relu'), &
    maxpool2d(pool_size=2), &
    flatten(), &
    dense(10, activation='softmax') &
  ])

  call net % print_info()

  epochs: do n = 1, num_epochs

    call net % train( &
      training_images, &
      label_digits(training_labels), &
      batch_size=128, &
      epochs=1, &
      optimizer=sgd(learning_rate=3.) &
    )

    if (this_image() == 1) &
      print '(a,i2,a,f5.2,a)', 'Epoch ', n, ' done, Accuracy: ', accuracy( &
        net, validation_images, label_digits(validation_labels)) * 100, ' %'

  end do epochs

  print '(a,f5.2,a)', 'Testing accuracy: ', &
    accuracy(net, testing_images, label_digits(testing_labels)) * 100, '%'

contains

  real function accuracy(net, x, y)
    type(network), intent(in out) :: net
    real, intent(in) :: x(:,:), y(:,:)
    integer :: i, good
    good = 0
    do i = 1, size(x, dim=2)
      if (all(maxloc(net % predict(x(:,i))) == maxloc(y(:,i)))) then
        good = good + 1
      end if
    end do
    accuracy = real(good) / size(x, dim=2)
  end function accuracy

end program cnn_mnist