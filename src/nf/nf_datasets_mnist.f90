module nf_datasets_mnist

  !! Procedures to work with MNIST dataset, usable with data format
  !! as provided in this repo and not the original data format (idx).

  implicit none

  private
  public :: label_digits, load_mnist

  interface

    pure module function label_digits(labels) result(res)
      !! Converts an array of individual MNIST labels (e.g. 3)
      !! into a form that can be used to evaluate against dense layer output,
      !! e.g. [0, 0, 0, 1, 0, 0, 0, 0, 0].
      implicit none
      real, intent(in) :: labels(:)
        !! Array of labels with single digit values in the range 0-9
      real :: res(10, size(labels))
        !! 10-element array of zeros and a single one indicating the digit
    end function label_digits
  
    module subroutine load_mnist(training_images, training_labels, &
                                 validation_images, validation_labels, &
                                 testing_images, testing_labels)
      !! Loads the MNIST dataset into arrays.
      implicit none
      real, allocatable, intent(in out) :: training_images(:,:)
      real, allocatable, intent(in out) :: training_labels(:)
      real, allocatable, intent(in out) :: validation_images(:,:)
      real, allocatable, intent(in out) :: validation_labels(:)
      real, allocatable, intent(in out), optional :: testing_images(:,:)
      real, allocatable, intent(in out), optional :: testing_labels(:)
    end subroutine load_mnist
  
  end interface

end module nf_datasets_mnist
