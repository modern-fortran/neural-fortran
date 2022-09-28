module nf_io_binary

  !! This module provides subroutines to read binary files using direct access.

  implicit none

  private
  public :: read_binary_file

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

end module nf_io_binary
