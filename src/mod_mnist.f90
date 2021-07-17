module mod_mnist

  ! Procedures to work with MNIST dataset, usable with data format
  ! as provided in this repo and not the original data format (idx).

  use iso_fortran_env, only: real32 ! TODO make MNIST work with arbitrary precision
  use mod_io, only: read_binary_file
  use mod_kinds, only: ik, rk

  implicit none

  private

  public :: label_digits, load_mnist, print_image

contains

  pure function digits(x)
    ! Returns an array of 10 reals, with zeros everywhere
    ! and a one corresponding to the input number, for example:
    !   digits(0) = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    !   digits(1) = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
    !   digits(6) = [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
    real(rk), intent(in) :: x
    real(rk) :: digits(10)
    digits = 0
    digits(int(x + 1)) = 1
  end function digits

  pure function label_digits(labels) result(res)
    ! Converts an array of MNIST labels into a form
    ! that can be input to the network_type instance.
    real(rk), intent(in) :: labels(:)
    real(rk) :: res(10, size(labels))
    integer(ik) :: i
    do i = 1, size(labels)
      res(:,i) = digits(labels(i))
    end do
  end function label_digits

  subroutine load_mnist(tr_images, tr_labels, te_images,&
                        te_labels, va_images, va_labels)
    ! Loads the MNIST dataset into arrays.
    real(rk), allocatable, intent(in out) :: tr_images(:,:), tr_labels(:)
    real(rk), allocatable, intent(in out) :: te_images(:,:), te_labels(:)
    real(rk), allocatable, intent(in out), optional :: va_images(:,:), va_labels(:)
    integer(ik), parameter :: dtype = 4, image_size = 784
    integer(ik), parameter :: tr_nimages = 50000
    integer(ik), parameter :: te_nimages = 10000
    integer(ik), parameter :: va_nimages = 10000

    call read_binary_file('../data/mnist/mnist_training_images.dat',&
                          dtype, image_size, tr_nimages, tr_images)
    call read_binary_file('../data/mnist/mnist_training_labels.dat',&
                          dtype, tr_nimages, tr_labels)

    call read_binary_file('../data/mnist/mnist_testing_images.dat',&
                          dtype, image_size, te_nimages, te_images)
    call read_binary_file('../data/mnist/mnist_testing_labels.dat',&
                          dtype, te_nimages, te_labels)

    if (present(va_images) .and. present(va_labels)) then
      call read_binary_file('../data/mnist/mnist_validation_images.dat',&
                            dtype, image_size, va_nimages, va_images)
      call read_binary_file('../data/mnist/mnist_validation_labels.dat',&
                            dtype, va_nimages, va_labels)
    end if

  end subroutine load_mnist

  subroutine print_image(images, labels, n)
    ! Prints a single image and label to screen.
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

end module mod_mnist
