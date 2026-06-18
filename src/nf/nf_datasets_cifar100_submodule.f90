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

    character(*), parameter :: train_file = 'cifar-100-binary/train.bin'
    character(*), parameter :: test_file = 'cifar-100-binary/test.bin'
    integer, parameter :: record_size = 3074
    integer :: num_records
    integer :: num_training_images
    integer :: num_validation_images
    logical :: file_exists

    ! Check if cifar100 data is present and download it if not.
    inquire(file=train_file, exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar100_url)
    num_records = cifar_record_count(train_file, record_size)
    call split_training_validation(num_records, num_training_images, &
      num_validation_images)

    if (present(testing_images) .neqv. present(testing_labels)) &
      error stop 'testing_images and testing_labels must be present together'

    if (present(testing_images)) then
      inquire(file=test_file, exist=file_exists)
      if (.not. file_exists) call download_and_unpack(cifar100_url)
    end if

    allocate(training_images(3072, num_training_images))
    allocate(training_labels(num_training_images))

    call read_cifar(train_file, num_records, training_images_dummy, &
      training_labels_dummy, .true.)

    training_images = training_images_dummy(:, 1:num_training_images)
    training_labels = training_labels_dummy(1:num_training_images)

    validation_images = training_images_dummy(:, &
      num_training_images + 1:num_records)
    validation_labels = training_labels_dummy(num_training_images + 1:num_records)

    if (present(testing_images)) then
      call read_cifar(test_file, cifar_record_count(test_file, record_size), &
        testing_images, testing_labels, .true.)
    end if

  end subroutine load_cifar100

  function cifar_record_count(filename, record_size) result(nrec)
    character(*), intent(in) :: filename
    integer, intent(in) :: record_size
    integer :: nrec
    integer :: file_size

    inquire(file=filename, size=file_size)
    if (file_size <= 0 .or. mod(file_size, record_size) /= 0) &
      error stop 'Invalid CIFAR file size'

    nrec = file_size / record_size
  end function cifar_record_count

  subroutine split_training_validation(num_records, num_training_images, &
      num_validation_images)
    integer, intent(in) :: num_records
    integer, intent(out) :: num_training_images
    integer, intent(out) :: num_validation_images

    if (num_records < 2) error stop 'CIFAR-100 train file must contain at least 2 records'

    num_validation_images = min(10000, max(1, num_records / 5))
    num_training_images = num_records - num_validation_images
  end subroutine split_training_validation

end submodule nf_datasets_cifar100_submodule
