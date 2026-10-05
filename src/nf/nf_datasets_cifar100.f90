module nf_datasets_cifar100

  !! Procedures to work with cifar100 dataset, usable with data format
  !! as provided in this repo and not the original data format (idx).

  implicit none

  private
  public :: label_digits_cifar100, load_cifar100

  interface

    pure module function label_digits_cifar100(labels) result(res)
      implicit none
      real, intent(in) :: labels(:)
        !! Array of labels with single digit values in the range 0-99
      real :: res(100, size(labels))
        !! 100-element array of zeros and a single one indicating the digit
    end function label_digits_cifar100
  
    module subroutine load_cifar100(training_images, training_images_dummy, &
                                 training_labels, training_labels_dummy, &
                                 validation_images, validation_labels, &
                                 testing_images, testing_labels)
      !! Loads the cifar100 dataset into arrays.
      implicit none

      real, allocatable, intent(in out) :: training_images(:,:), training_images_dummy(:,:)
      real, allocatable, intent(in out) :: training_labels(:), training_labels_dummy(:)
      real, allocatable, intent(in out) :: validation_images(:,:)
      real, allocatable, intent(in out) :: validation_labels(:)
      real, allocatable, intent(in out), optional :: testing_images(:,:)
      real, allocatable, intent(in out), optional :: testing_labels(:)
    end subroutine load_cifar100
  
  end interface

end module nf_datasets_cifar100