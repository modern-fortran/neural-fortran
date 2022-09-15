program dense_from_keras

  ! This example demonstrates loading a dense model
  ! pre-trained on the MNIST dataset from a Keras HDF5
  ! file and running an inferrence on the testing dataset.

  use nf, only: network, label_digits, load_mnist
  use nf_datasets, only: download_and_unpack, keras_dense_mnist_url

  implicit none

  type(network) :: net
  real, allocatable :: training_images(:,:), training_labels(:)
  real, allocatable :: validation_images(:,:), validation_labels(:)
  real, allocatable :: testing_images(:,:), testing_labels(:)
  character(*), parameter :: keras_dense_path = 'keras_dense_mnist.h5'
  logical :: file_exists

  inquire(file=keras_dense_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_dense_mnist_url)

  call load_mnist(training_images, training_labels, &
                  validation_images, validation_labels, &
                  testing_images, testing_labels)

  print '("Loading a pre-trained dense model from Keras")'
  print '(60("="))'

  net = network(keras_dense_path)

  call net % print_info()

  if (this_image() == 1) &
    print '(a,f5.2,a)', 'Accuracy: ', accuracy( &
      net, testing_images, label_digits(testing_labels)) * 100, ' %'

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

end program dense_from_keras
