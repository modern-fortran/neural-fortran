submodule(nf_datasets_cifar10) nf_datasets_cifar10_submodule

  use nf_datasets, only: download_and_unpack, cifar10_url
  use nf_io_binary, only: read_binary_file, read_cifar

  implicit none

  integer, parameter :: message_len = 128

contains

  module subroutine load_cifar10(training_images, training_images_dummy, &
                                 training_labels, training_labels_dummy, &
                                 validation_images, validation_labels, &
                                 testing_images, testing_labels)

    real, allocatable, intent(in out) :: training_images_dummy(:,:), training_images(:,:)
    real, allocatable, intent(in out) :: training_labels(:), training_labels_dummy(:)

    real, allocatable, intent(in out) :: validation_images(:,:)
    real, allocatable, intent(in out) :: validation_labels(:)

    real, allocatable, intent(in out), optional :: testing_images(:,:)
    real, allocatable, intent(in out), optional :: testing_labels(:)

    character(*), parameter :: batch_files(5) = [ &
      'cifar-10-batches-bin/data_batch_1.bin', &
      'cifar-10-batches-bin/data_batch_2.bin', &
      'cifar-10-batches-bin/data_batch_3.bin', &
      'cifar-10-batches-bin/data_batch_4.bin', &
      'cifar-10-batches-bin/data_batch_5.bin' &
    ]
    character(*), parameter :: test_file = 'cifar-10-batches-bin/test_batch.bin'
    integer, parameter :: record_size = 3073
    integer :: batch_counts(5)
    integer :: batch
    integer :: offset
    logical :: file_exists

    ! Check if cifar10 data is present and download it if not.
    do batch = 1, size(batch_files)
      inquire(file=batch_files(batch), exist=file_exists)
      if (.not. file_exists) call download_and_unpack(cifar10_url)
      batch_counts(batch) = cifar_record_count(batch_files(batch), record_size)
    end do

    if (present(testing_images) .neqv. present(testing_labels)) &
      error stop 'testing_images and testing_labels must be present together'

    if (present(testing_images)) then
      inquire(file=test_file, exist=file_exists)
      if (.not. file_exists) call download_and_unpack(cifar10_url)
    end if

    offset = 0

    allocate(training_images(3072, sum(batch_counts(1:4))))
    allocate(training_labels(sum(batch_counts(1:4))))

    do batch = 1, 4
      call read_cifar(batch_files(batch), batch_counts(batch), &
      training_images_dummy, training_labels_dummy, .false.)

      training_images(:,(offset+1):(offset+batch_counts(batch))) = training_images_dummy
      training_labels(offset+1:offset+batch_counts(batch)) = training_labels_dummy
      offset = offset + batch_counts(batch)
    end do

    call read_cifar(batch_files(5), batch_counts(5), &
    validation_images, validation_labels, .false.)

    if (present(testing_images)) then
      call read_cifar(test_file, cifar_record_count(test_file, record_size), &
      testing_images, testing_labels, .false.)
    end if

  end subroutine load_cifar10

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

end submodule nf_datasets_cifar10_submodule
