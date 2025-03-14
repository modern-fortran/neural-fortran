program cnn_mnist_1d

    use nf, only: network, sgd, &
      input, conv1d, maxpool1d, flatten, dense, reshape, locally_connected1d, &
      load_mnist, label_digits, softmax, relu
  
    implicit none
  
    type(network) :: net
  
    real, allocatable :: training_images(:,:), training_labels(:)
    real, allocatable :: validation_images(:,:), validation_labels(:)
    real, allocatable :: testing_images(:,:), testing_labels(:)
    integer :: n
    integer, parameter :: num_epochs = 250
  
    call load_mnist(training_images, training_labels, &
                    validation_images, validation_labels, &
                    testing_images, testing_labels)
  
    net = network([ &
      input(784), &
      reshape(28, 28), &
      locally_connected1d(filters=8, kernel_size=3, activation=relu()), &
      maxpool1d(pool_size=2), &
      locally_connected1d(filters=16, kernel_size=3, activation=relu()), &
      maxpool1d(pool_size=2), &
      dense(10, activation=softmax()) &
    ])
  
    call net % print_info()
  
    epochs: do n = 1, num_epochs
  
      call net % train( &
        training_images, &
        label_digits(training_labels), &
        batch_size=16, &
        epochs=1, &
        optimizer=sgd(learning_rate=0.01) &
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
  
  end program cnn_mnist_1d
  