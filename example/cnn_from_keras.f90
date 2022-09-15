program cnn_from_keras

  ! This example demonstrates loading a convolutional model
  ! pre-trained on the MNIST dataset from a Keras HDF5
  ! file and running an inferrence on the testing dataset.

  use nf, only: network, label_digits, load_mnist
  use nf_datasets, only: download_and_unpack, keras_cnn_mnist_url

  implicit none

  type(network) :: net
  real, allocatable :: training_images(:,:), training_labels(:)
  real, allocatable :: validation_images(:,:), validation_labels(:)
  real, allocatable :: testing_images(:,:), testing_labels(:)
  character(*), parameter :: keras_cnn_path = 'keras_cnn_mnist.h5'
  logical :: file_exists
  real :: acc

  inquire(file=keras_cnn_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_cnn_mnist_url)

  call load_mnist(training_images, training_labels, &
                  validation_images, validation_labels, &
                  testing_images, testing_labels)

  print '("Loading a pre-trained CNN model from Keras")'
  print '(60("="))'

  net = network(keras_cnn_path)

  call net % print_info()

  if (this_image() == 1) then
    acc = accuracy( &
      net, &
      reshape(testing_images(:,:), shape=[1,28,28,size(testing_images,2)]), &
      label_digits(testing_labels) &
    )
    print '(a,f5.2,a)', 'Accuracy: ', acc * 100, ' %'
  end if

contains

  real function accuracy(net, x, y)
    type(network), intent(in out) :: net
    real, intent(in) :: x(:,:,:,:), y(:,:)
    integer :: i, good
    good = 0
    do i = 1, size(x, dim=4)
      if (all(maxloc(net % predict(x(:,:,:,i))) == maxloc(y(:,i)))) then
        good = good + 1
      end if
    end do
    accuracy = real(good) / size(x, dim=4)
  end function accuracy

end program cnn_from_keras
