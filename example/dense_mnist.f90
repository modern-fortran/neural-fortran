program dense_mnist

  use nf, only: dense, input, network, sgd, label_digits, load_mnist, corr

  implicit none

  type(network) :: net
  real, allocatable :: training_images(:,:), training_labels(:)
  real, allocatable :: validation_images(:,:), validation_labels(:)
  integer :: n, num_epochs

  call load_mnist(training_images, training_labels, &
                  validation_images, validation_labels)

  print '("MNIST")'
  print '(60("="))'

  net = network([ &
    input(784), &
    dense(30), &
    dense(10) &
  ])
  num_epochs = 10

  call net % print_info()

  if (this_image() == 1) &
    print '(a,f5.2,a)', 'Initial accuracy: ', accuracy( &
      net, validation_images, label_digits(validation_labels)) * 100, ' %'

  epochs: do n = 1, num_epochs

    call net % train( &
      training_images, &
      label_digits(training_labels), &
      batch_size=100, &
      epochs=1, &
      optimizer=sgd(learning_rate=3.) &
    )

    if (this_image() == 1) &
      print '(a,i2,a,f5.2,a)', 'Epoch ', n, ' done, Accuracy: ', accuracy( &
        net, validation_images, label_digits(validation_labels)) * 100, ' %'

    block
      real, allocatable :: output_metrics(:,:) ! 2 metrics; 1st is default loss function (quadratic), other is Pearson corr.
      output_metrics = net % evaluate(validation_images, label_digits(validation_labels), metric=corr())
      print *, "Metrics: quadratic loss, Pearson corr.:", sum(output_metrics, 1) / size(output_metrics, 1)
    end block

    block
      real, allocatable :: output_metrics(:,:) ! 3 metrics; 1st is default loss function (quadratic), others are Pearson corr.
      output_metrics = net % evaluate(validation_images, label_digits(validation_labels), metrics=[corr(), corr()])
      print *, "Metrics: quadratic loss, Pearson corr.:", sum(output_metrics, 1) / size(output_metrics, 1)
    end block

  end do epochs

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

end program dense_mnist
