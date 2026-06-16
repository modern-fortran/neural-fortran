program cifar100

  use nf, only: network, sgd, &
    input, conv, maxpool, flatten, dense, reshape, &
    load_cifar100, label_digits_cifar100, softmax, relu
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

  call load_cifar100(training_images, training_images_dummy, &
                     training_labels, training_labels_dummy, &
                     validation_images, validation_labels, &
                     testing_images, testing_labels)
  
  print *, 'Training images shape: ', shape(training_images)
  print *, 'Training labels shape: ', shape(training_labels)
  print *, 'Validation images shape: ', shape(validation_images)
  print *, 'Validation labels shape: ', shape(validation_labels)
  print *, 'Testing images shape: ', shape(testing_images)
  print *, 'Testing labels shape: ', shape(testing_labels)
  print *, 'maxval', maxval(training_labels), 'minval', minval(training_labels)
    
    net = network([ &
        input(3072), &
        reshape(3, 32, 32), &
        conv(filters=8, kernel_width=3, kernel_height=3, activation=relu()), &
        maxpool(pool_width=2, pool_height=2, stride=2), &
        conv(filters=16, kernel_width=3, kernel_height=3, activation=relu()), &
        maxpool(pool_width=2, pool_height=2, stride=2), &
        dense(100, activation=softmax()) &
    ])

  call net % print_info()

  epochs: do n = 1, num_epochs

    call net % train( &
      training_images(:, 1:1000), &
      label_digits_cifar100(training_labels(1:1000)), &
      batch_size=16, &
      epochs=1, &
      optimizer=sgd(learning_rate=0.001) &
    )

    print '(a,i2,a,f5.2,a)', 'Epoch ', n, ' done, Accuracy: ', accuracy( &
      net, validation_images, label_digits_cifar100(validation_labels)) * 100, ' %'

  end do epochs

  print '(a,f5.2,a)', 'Testing accuracy: ', &
    accuracy(net, testing_images, label_digits_cifar100(testing_labels)) * 100, '%'

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




end program cifar100

 