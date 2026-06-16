module nf_datasets_cifar10

  !! Procedures to work with cifar10 dataset, usable with data format
  !! as provided in this repo and not the original data format (idx).

  implicit none

  private
  public :: load_cifar10

  interface
  
    module subroutine load_cifar10(training_images, training_images_dummy, &
                                 training_labels, training_labels_dummy, &
                                 validation_images, validation_labels, &
                                 testing_images, testing_labels)
      !! Loads the cifar10 dataset into arrays.
      implicit none

      real, allocatable, intent(in out) :: training_images(:,:), training_images_dummy(:,:)
      real, allocatable, intent(in out) :: training_labels(:), training_labels_dummy(:)
      real, allocatable, intent(in out) :: validation_images(:,:)
      real, allocatable, intent(in out) :: validation_labels(:)
      real, allocatable, intent(in out), optional :: testing_images(:,:)
      real, allocatable, intent(in out), optional :: testing_labels(:)
    end subroutine load_cifar10
  
  end interface

end module nf_datasets_cifar10
