module nf_io_binary

  !! This module provides subroutines to read binary files using direct access.

  implicit none

  private
  public :: read_binary_file, read_cifar10, read_cifar100

  interface read_binary_file

    module subroutine read_binary_file_1d(filename, dtype, nrec, array)
      !! Read a binary file into a 1-d real array using direct access.
      implicit none
      character(*), intent(in) :: filename
        !! Path to the file to read
      integer, intent(in) :: dtype
        !! Number of bytes per element
      integer, intent(in) :: nrec
        !! Number of records to read
      real, allocatable, intent(in out) :: array(:)
        !! Array to store the data in
    end subroutine read_binary_file_1d

    module subroutine read_binary_file_2d(filename, dtype, dsize, nrec, array)
      !! Read a binary file into a 2-d real array using direct access.
      implicit none
      character(*), intent(in) :: filename
        !! Path to the file to read
      integer, intent(in) :: dtype
        !! Number of bytes per element
      integer, intent(in) :: dsize
        !! Number of elements in a record
      integer, intent(in) :: nrec
        !! Number of records to read
      real, allocatable, intent(in out) :: array(:,:)
        !! Array to store the data in
    end subroutine read_binary_file_2d

  end interface read_binary_file

  interface read_cifar10
    module subroutine read_cifar10(filename, nrec, images, labels)
      !! Read a CIFAR-10 binary file into a 2-d integer(1) array.
      implicit none
      character(*), intent(in) :: filename
        !! Path to the CIFAR-10 binary file to read
      integer, intent(in) :: nrec
        !! Number of records to read (e.g. 10000 for CIFAR-10)
      real, allocatable, intent(out) :: images(:,:)
        !! Array to store the image data in (should be dimensioned 3 x 32 x 32 x nrec)
      real, allocatable, intent(out) :: labels(:)
        !! Array to store the labels in (should be dimensioned nrec)
    end subroutine read_cifar10
  end interface read_cifar10

  interface read_cifar100
    module subroutine read_cifar100(filename, nrec, images, labels)
      !! Read a CIFAR-100 binary file into a 2-d integer(1) array.
      implicit none
      character(*), intent(in) :: filename
        !! Path to the CIFAR-100 binary file to read
      integer, intent(in) :: nrec
        !! Number of records to read (e.g. 10000 for CIFAR-100)
      real, allocatable, intent(out) :: images(:,:)
        !! Array to store the image data in (should be dimensioned 3 x 32 x 32 x nrec)
      real, allocatable, intent(out) :: labels(:)
        !! Array to store the labels in (should be dimensioned nrec)
    end subroutine read_cifar100
  end interface read_cifar100

  interface read_cifar_common
    module subroutine read_cifar_common(filename, nrec, images, labels, cifar_100)
      !! Internal helper for CIFAR readers.
      implicit none
      character(*), intent(in) :: filename
      integer, intent(in) :: nrec
      real, allocatable, intent(out) :: images(:,:)
      real, allocatable, intent(out) :: labels(:)
      logical, intent(in) :: cifar_100
    end subroutine read_cifar_common
  end interface read_cifar_common

end module nf_io_binary
