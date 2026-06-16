module nf_io_binary

  !! This module provides subroutines to read binary files using direct access.

  implicit none

  private
  public :: read_binary_file, read_cifar

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

  interface read_cifar
    module subroutine read_cifar(filename, nrec, images, labels, cifar_100)
      !! Read a CIFAR binary file into a 2-d integer(1) array.
      implicit none
      character(*), intent(in) :: filename
        !! Path to the CIFAR binary file to read
      integer, intent(in) :: nrec
        !! Number of records to read (e.g. 10000 for CIFAR-10)
      real, allocatable, intent(out) :: images(:,:)
        !! Array to store the image data in (should be dimensioned 3 x 32 x 32 x nrec)
      real, allocatable, intent(out) :: labels(:)
        !! Array to store the labels in (should be dimensioned nrec)
      logical, intent(in) :: cifar_100
        !! Set to true if reading CIFAR-100 data (default is false, i.e. CIFAR-10)
    end subroutine read_cifar
  end interface read_cifar

end module nf_io_binary
