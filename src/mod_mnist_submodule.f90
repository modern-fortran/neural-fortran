submodule(mod_mnist) mod_mnist_submodule

  !! Procedures to work with MNIST dataset, usable with data format
  !! as provided in this repo and not the original data format (idx).

  ! TODO make MNIST work with arbitrary precision

  use mod_io, only: read_binary_file
  use mod_kinds, only: ik, rk

  implicit none

  integer, parameter :: message_len = 128

contains

  subroutine download_and_uncompress()
    character(len=*), parameter :: download_mechanism = 'curl -LO '
    character(len=*), parameter :: base_url='https://github.com/modern-fortran/neural-fortran/files/8498876/'
    character(len=*), parameter :: download_filename = 'mnist.tar.gz'
    character(len=*), parameter :: download_command = download_mechanism // base_url // download_filename
    character(len=*), parameter :: uncompress_file = 'tar xvzf ' // download_filename
    character(len=message_len) :: command_message
    character(len=:), allocatable :: error_message
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
    real(rk), intent(in) :: labels(:)
    real(rk) :: res(10, size(labels))
    integer(ik) :: i
    do i = 1, size(labels)
      res(:,i) = digits(labels(i))
    end do
  contains
    pure function digits(x)
      !! Returns an array of 10 reals, with zeros everywhere
      !! and a one corresponding to the input number, for example:
      !!   digits(0) = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
      !!   digits(1) = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
      !!   digits(6) = [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
      real(rk), intent(in) :: x
      real(rk) :: digits(10)
      digits = 0
      digits(int(x + 1)) = 1
    end function digits
  end function label_digits

  module subroutine load_mnist(tr_images, tr_labels, te_images,&
                        te_labels, va_images, va_labels)
    real(rk), allocatable, intent(in out) :: tr_images(:,:), tr_labels(:)
    real(rk), allocatable, intent(in out) :: te_images(:,:), te_labels(:)
    real(rk), allocatable, intent(in out), optional :: va_images(:,:), va_labels(:)
    integer(ik), parameter :: dtype = 4, image_size = 784
    integer(ik), parameter :: tr_nimages = 50000
    integer(ik), parameter :: te_nimages = 10000
    integer(ik), parameter :: va_nimages = 10000
    logical :: file_exists

    ! Check if MNIST data is present and download it if not.
    inquire(file='mnist_training_images.dat', exist=file_exists)
    if (.not. file_exists) call download_and_uncompress()

    call read_binary_file('mnist_training_images.dat',&
                          dtype, image_size, tr_nimages, tr_images)
    call read_binary_file('mnist_training_labels.dat',&
                          dtype, tr_nimages, tr_labels)

    call read_binary_file('mnist_testing_images.dat',&
                          dtype, image_size, te_nimages, te_images)
    call read_binary_file('mnist_testing_labels.dat',&
                          dtype, te_nimages, te_labels)

    if (present(va_images) .and. present(va_labels)) then
      call read_binary_file('mnist_validation_images.dat',&
                            dtype, image_size, va_nimages, va_images)
      call read_binary_file('mnist_validation_labels.dat',&
                            dtype, va_nimages, va_labels)
    end if

  end subroutine load_mnist

  module subroutine print_image(images, labels, n)
    real(rk), intent(in) :: images(:,:), labels(:)
    integer(ik), intent(in) :: n
    real(rk) :: image(28, 28)
    character(len=1) :: char_image(28, 28)
    integer(ik) i, j
    image = reshape(images(:,n), [28, 28])
    char_image = '.'
    where (image > 0) char_image = '#'
    print *, labels(n)
    do j = 1, 28
      print *, char_image(:,j)
    end do
  end subroutine print_image

end submodule mod_mnist_submodule
