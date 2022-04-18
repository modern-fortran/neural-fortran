submodule(mod_io) mod_io_submodule

  implicit none

  integer, parameter :: message_len = 128

contains

  module subroutine read_binary_file_1d(filename, dtype, nrec, array)
    character(len=*), intent(in) :: filename
    integer(ik), intent(in) :: dtype, nrec
    real(rk), allocatable, intent(in out) :: array(:)
    integer(ik) :: fileunit
    character(len=message_len) :: io_message
    integer :: io_status
    io_status = 0
    open(newunit=fileunit, file=filename, access='direct', action='read', &
      recl=dtype * nrec, status='old', iostat=io_status, iomsg=io_message)
    if (io_status /= 0) error stop trim(io_message)
    allocate(array(nrec))
    read(fileunit, rec=1) array
    close(fileunit)
  end subroutine read_binary_file_1d

  module subroutine read_binary_file_2d(filename, dtype, dsize, nrec, array)
    character(len=*), intent(in) :: filename
    integer(ik), intent(in) :: dtype, dsize, nrec
    real(rk), allocatable, intent(in out) :: array(:,:)
    integer(ik) :: fileunit, i
    character(len=message_len) :: io_message
    integer :: io_status
    io_status = 0
    open(newunit=fileunit, file=filename, access='direct', action='read', &
      recl=dtype * dsize, status='old', iostat=io_status, iomsg=io_message)
    if (io_status /= 0) error stop trim(io_message)
    allocate(array(dsize, nrec))
    do i = 1, nrec
      read(fileunit, rec=i) array(:,i)
    end do
    close(fileunit)
  end subroutine read_binary_file_2d

end submodule mod_io_submodule
