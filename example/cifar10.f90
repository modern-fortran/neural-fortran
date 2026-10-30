program cifar10

  use nf, only: network, sgd, &
    input, conv, maxpool, flatten, dense, reshape, &
    load_cifar10, label_digits, softmax, relu

  implicit none

  type(network) :: net

  real, allocatable :: training_images(:,:), training_images_dummy(:,:)
  real, allocatable :: training_labels(:), training_labels_dummy(:)

  real, allocatable :: validation_images(:,:)
  real, allocatable :: validation_labels(:)
  real, allocatable :: testing_images(:,:)
  real, allocatable :: testing_labels(:)
  integer :: n
  integer, parameter :: num_epochs = 250

  call load_cifar10(training_images, training_images_dummy, &
                     training_labels, training_labels_dummy, &
                     validation_images, validation_labels, &
                     testing_images, testing_labels)
    
    net = network([ &
        input(3072), &
        reshape(3, 32, 32), &
        conv(filters=8, kernel_width=3, kernel_height=3, activation=relu()), &
        maxpool(pool_width=2, pool_height=2, stride=2), &
        conv(filters=16, kernel_width=3, kernel_height=3, activation=relu()), &
        maxpool(pool_width=2, pool_height=2, stride=2), &
        dense(10, activation=softmax()) &
    ])

  call net % print_info()

  epochs: do n = 1, num_epochs

    call net % train( &
      training_images(:, 1:10000), &
      label_digits(training_labels(1:10000)), &
      batch_size=16, &
      epochs=1, &
      optimizer=sgd(learning_rate=0.001) &
    )

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




end program cifar10

 