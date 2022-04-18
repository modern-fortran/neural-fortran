module mod_mnist

  !! Procedures to work with MNIST dataset, usable with data format
  !! as provided in this repo and not the original data format (idx).

  use mod_kinds, only: ik, rk

  implicit none

  private

  public :: label_digits, load_mnist, print_image

  interface
  
    pure module function label_digits(labels) result(res)
      !! Converts an array of MNIST labels into a form
      !! that can be input to the network_type instance.
      implicit none
      real(rk), intent(in) :: labels(:)
      real(rk) :: res(10, size(labels))
    end function label_digits
  
    module subroutine load_mnist(tr_images, tr_labels, te_images,&
  
                          te_labels, va_images, va_labels)
      !! Loads the MNIST dataset into arrays.
      implicit none
      real(rk), allocatable, intent(in out) :: tr_images(:,:), tr_labels(:)
      real(rk), allocatable, intent(in out) :: te_images(:,:), te_labels(:)
      real(rk), allocatable, intent(in out), optional :: va_images(:,:), va_labels(:)
    end subroutine load_mnist
  
    module subroutine print_image(images, labels, n)
      !! Prints a single image and label to screen.
      implicit none
      real(rk), intent(in) :: images(:,:), labels(:)
      integer(ik), intent(in) :: n
    end subroutine print_image
  
  end interface

end module mod_mnist
