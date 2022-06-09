submodule(nf_io_binary) nf_io_binary_submodule

  implicit none

  integer, parameter :: message_len = 128

contains

  module subroutine read_binary_file_1d(filename, dtype, nrec, array)
    character(*), intent(in) :: filename
    integer, intent(in) :: dtype, nrec
    real, allocatable, intent(in out) :: array(:)
    integer :: fileunit
    character(message_len) :: io_message
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
    character(*), intent(in) :: filename
    integer, intent(in) :: dtype, dsize, nrec
    real, allocatable, intent(in out) :: array(:,:)
    integer :: fileunit, i
    character(message_len) :: io_message
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

end submodule nf_io_binary_submodule
