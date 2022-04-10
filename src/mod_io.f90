module mod_io

  use mod_kinds, only: ik, rk

  implicit none

  private

  public :: read_binary_file

  interface read_binary_file

    module subroutine read_binary_file_1d(filename, dtype, nrec, array)
      implicit none
      character(len=*), intent(in) :: filename
      integer(ik), intent(in) :: dtype, nrec
      real(rk), allocatable, intent(in out) :: array(:)
    end subroutine read_binary_file_1d

    module subroutine read_binary_file_2d(filename, dtype, dsize, nrec, array)
      implicit none
      character(len=*), intent(in) :: filename
      integer(ik), intent(in) :: dtype, dsize, nrec
      real(rk), allocatable, intent(in out) :: array(:,:)
    end subroutine read_binary_file_2d

  end interface read_binary_file

end module mod_io
