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

    integer, parameter :: dtype = 1, image_size = 3073
    integer, parameter :: num_training_images = 10000
    integer, parameter :: num_validation_images = 10000
    integer, parameter :: num_testing_images = 10000
    integer, parameter :: batch_size = 10000
    integer :: offset = 0
    logical :: file_exists

    ! Check if cifar10 data is present and download it if not.
    inquire(file='cifar-10-batches-bin/data_batch_1.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar10_url)
    inquire(file='cifar-10-batches-bin/data_batch_2.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar10_url)
    inquire(file='cifar-10-batches-bin/data_batch_3.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar10_url)
    inquire(file='cifar-10-batches-bin/data_batch_4.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar10_url)
    inquire(file='cifar-10-batches-bin/data_batch_5.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar10_url)
    inquire(file='cifar-10-batches-bin/test_batch.bin', exist=file_exists)
    if (.not. file_exists) call download_and_unpack(cifar10_url)

    allocate(training_images(3072,40000))
    allocate(training_labels(40000))

    call read_cifar('cifar-10-batches-bin/data_batch_1.bin', &
    num_training_images, training_images_dummy, training_labels_dummy, .false.)

    training_images(:,(offset+1):(offset+10000)) = training_images_dummy
    training_labels(offset+1:offset+10000) = training_labels_dummy(1:10000)
    offset = offset + 10000

    call read_cifar('cifar-10-batches-bin/data_batch_2.bin', &
    num_training_images, training_images_dummy, training_labels_dummy, .false.)

    training_images(:,(offset+1):(offset+10000)) = training_images_dummy
    training_labels(offset+1:offset+10000) = training_labels_dummy(1:10000)
    offset = offset + 10000

    call read_cifar('cifar-10-batches-bin/data_batch_3.bin', &
    num_training_images, training_images_dummy, training_labels_dummy, .false.)

    training_images(:,(offset+1):(offset+10000)) = training_images_dummy
    training_labels(offset+1:offset+10000) = training_labels_dummy(1:10000)
    offset = offset + 10000

    call read_cifar('cifar-10-batches-bin/data_batch_4.bin', &
    num_training_images, training_images_dummy, training_labels_dummy, .false.)

    training_images(:,(offset+1):(offset+10000)) = training_images_dummy
    training_labels(offset+1:offset+10000) = training_labels_dummy(1:10000)
    offset = offset + 10000

    call read_cifar('cifar-10-batches-bin/data_batch_5.bin', &
    num_training_images, validation_images, validation_labels, .false.)

    call read_cifar('cifar-10-batches-bin/test_batch.bin', &
    num_testing_images, testing_images, testing_labels, .false.)

  end subroutine load_cifar10

end submodule nf_datasets_cifar10_submodule
