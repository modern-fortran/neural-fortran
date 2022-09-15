program test_cnn_from_keras

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: network
  use nf_datasets, only: download_and_unpack, keras_cnn_mnist_url

  implicit none

  type(network) :: net
  character(*), parameter :: test_data_path = 'keras_cnn_mnist.h5'
  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_cnn_mnist_url)

  net = network(test_data_path)

  block

    use nf, only: load_mnist, label_digits

    real, allocatable :: training_images(:,:), training_labels(:)
    real, allocatable :: validation_images(:,:), validation_labels(:)
    real, allocatable :: testing_images(:,:), testing_labels(:)
    real, allocatable :: input_reshaped(:,:,:,:)
    real :: acc

    call load_mnist(training_images, training_labels, &
                    validation_images, validation_labels, &
                    testing_images, testing_labels)

    ! Use only the first 1000 images to make the test short
    input_reshaped = reshape(testing_images(:,:1000), shape=[1,28,28,1000])

    acc = accuracy(net, input_reshaped, label_digits(testing_labels(:1000)))

    if (acc < 0.97) then
      write(stderr, '(a)') &
        'Pre-trained network accuracy should be > 0.97.. failed'
      ok = .false.
    end if

  end block

  if (ok) then
    print '(a)', 'test_cnn_from_keras: All tests passed.'
  else
    write(stderr, '(a)') &
      'test_cnn_from_keras: One or more tests failed.'
    stop 1
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

end program test_cnn_from_keras
