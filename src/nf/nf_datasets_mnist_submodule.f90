submodule(nf_datasets_mnist) nf_datasets_mnist_submodule

  use nf_io, only: read_binary_file

  implicit none

  integer, parameter :: message_len = 128

contains

  subroutine download_and_uncompress()
    character(*), parameter :: download_mechanism = 'curl -LO '
    character(*), parameter :: base_url='https://github.com/modern-fortran/neural-fortran/files/8498876/'
    character(*), parameter :: download_filename = 'mnist.tar.gz'
    character(*), parameter :: download_command = download_mechanism // base_url // download_filename
    character(*), parameter :: uncompress_file = 'tar xvzf ' // download_filename
    character(message_len) :: command_message
    character(:), allocatable :: error_message
    integer :: exit_status, command_status

    exit_status=0
    call execute_command_line(command=download_command, wait=.true., &
      exitstat=exit_status, cmdstat=command_status, cmdmsg=command_message)

    if (any([exit_status, command_status] /= 0)) then
      error_message = 'command "' // download_command // '" failed'
      if (command_status /= 0) error_message = error_message // " with message " // trim(command_message)
      error stop error_message
    end if

    call execute_command_line(command=uncompress_file, wait=.true., &
      exitstat=exit_status, cmdstat=command_status, cmdmsg=command_message)

    if (any([exit_status, command_status] /= 0)) then
      error_message = 'command "' // uncompress_file // '" failed'
      if (command_status /= 0) error_message = error_message // " with message " // trim(command_message)
      error stop  error_message
    end if

  end subroutine download_and_uncompress


  pure module function label_digits(labels) result(res)
    real, intent(in) :: labels(:)
    real :: res(10, size(labels))
    integer :: i
    do i = 1, size(labels)
      res(:,i) = digits(labels(i))
    end do
  contains
    pure function digits(x)
      !! Returns an array of 10 reals, with zeros everywhere
      !! and a one corresponding to the input digit.
      !!
      !! Example
      !!
      !! ```
      !! digits(0) = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
      !! digits(1) = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
      !! digits(6) = [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
      !! ```
      real, intent(in) :: x
        !! Input digit (0-9)
      real :: digits(10)
        !! 10-element array of zeros with a single one
        !! indicating the input digit
      digits = 0
      digits(int(x + 1)) = 1
    end function digits
  end function label_digits


  module subroutine load_mnist(training_images, training_labels, &
                               validation_images, validation_labels, &
                               testing_images, testing_labels)
    real, allocatable, intent(in out) :: training_images(:,:)
    real, allocatable, intent(in out) :: training_labels(:)
    real, allocatable, intent(in out) :: validation_images(:,:)
    real, allocatable, intent(in out) :: validation_labels(:)
    real, allocatable, intent(in out), optional :: testing_images(:,:)
    real, allocatable, intent(in out), optional :: testing_labels(:)

    integer, parameter :: dtype = 4, image_size = 784
    integer, parameter :: num_training_images = 50000
    integer, parameter :: num_validation_images = 10000
    integer, parameter :: num_testing_images = 10000
    logical :: file_exists

    ! Check if MNIST data is present and download it if not.
    inquire(file='mnist_training_images.dat', exist=file_exists)
    if (.not. file_exists) call download_and_uncompress()

    ! Load the training dataset (50000 samples)
    call read_binary_file('mnist_training_images.dat', &
      dtype, image_size, num_training_images, training_images)
    call read_binary_file('mnist_training_labels.dat', &
      dtype, num_training_images, training_labels)

    ! Load the validation dataset (10000 samples), for use while training
    call read_binary_file('mnist_validation_images.dat', &
      dtype, image_size, num_validation_images, validation_images)
    call read_binary_file('mnist_validation_labels.dat', &
      dtype, num_validation_images, validation_labels)

    ! Load the testing dataset (10000 samples), to test after training
    if (present(testing_images) .and. present(testing_labels)) then
      call read_binary_file('mnist_testing_images.dat', &
        dtype, image_size, num_testing_images, testing_images)
      call read_binary_file('mnist_testing_labels.dat', &
        dtype, num_testing_images, testing_labels)
    end if

  end subroutine load_mnist


  module subroutine print_image(images, labels, n)
    real, intent(in) :: images(:,:)
    real, intent(in) :: labels(:)
    integer, intent(in) :: n
    real :: image(28, 28)
    character :: char_image(28, 28)
    integer i, j
    image = reshape(images(:,n), [28, 28])
    char_image = '.'
    where (image > 0) char_image = '#'
    print *, labels(n)
    do j = 1, 28
      print *, char_image(:,j)
    end do
  end subroutine print_image

end submodule nf_datasets_mnist_submodule
