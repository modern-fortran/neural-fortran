submodule(nf_datasets_cifar100) nf_datasets_cifar100_submodule

  use nf_datasets, only: download_and_unpack, cifar100_url
  use nf_io_binary, only: read_binary_file, read_cifar

  implicit none

  integer, parameter :: message_len = 128

contains

  pure module function label_digits_cifar100(labels) result(res)
    real, intent(in) :: labels(:)
    real :: res(100, size(labels))
    integer :: i
    do i = 1, size(labels)
      res(:,i) = digits(labels(i))
    end do
  contains
    pure function digits(x)
      !! Returns an array of 100 reals, with zeros everywhere
      !! and a one corresponding to the input digit.
      !!
      real, intent(in) :: x
        !! Input digit (0-99)
      real :: digits(100)
        !! 100-element array of zeros with a single one
        !! indicating the input digit
      digits = 0
      digits(int(x + 1)) = 1
    end function digits
  end function label_digits_cifar100


  module subroutine load_cifar100(training_images, training_images_dummy, &
                                 training_labels, training_labels_dummy, &
                                 validation_images, validation_labels, &
                                 testing_images, testing_labels)

    real, allocatable, intent(in out) :: training_images_dummy(:,:), training_images(:,:)
    real, allocatable, intent(in out) :: training_labels(:), training_labels_dummy(:)

    real, allocatable, intent(in out) :: validation_images(:,:)
    real, allocatable, intent(in out) :: validation_labels(:)

    real, allocatable, intent(in out), optional :: testing_images(:,:)
    real, allocatable, intent(in out), optional :: testing_labels(:)

    integer, parameter :: dtype = 1, image_size = 3073
    integer, parameter :: num_training_images = 50000
    integer, parameter :: num_validation_images = 10000
    integer, parameter :: num_testing_images = 10000
    integer, parameter :: batch_size = 10000
    logical :: file_exists

    ! Check if cifar100 data is present and download it if not.
    inquire(file='cifar-100-binary/train.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar100_url)
    inquire(file='cifar-100-binary/test.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar100_url)

    allocate(training_images(3072,40000))
    allocate(training_labels(40000))

    call read_cifar('cifar-100-binary/train.bin', &
    num_training_images, training_images_dummy, training_labels_dummy, .true.)

    training_images = training_images_dummy(:, 1:40000)
    training_labels = training_labels_dummy(1:40000)

    validation_images = training_images_dummy(:, 40001:50000)
    validation_labels = training_labels_dummy(40001:50000)

    call read_cifar('cifar-100-binary/test.bin', &
    num_testing_images, testing_images, testing_labels, .true.)

  end subroutine load_cifar100

end submodule nf_datasets_cifar100_submodule
