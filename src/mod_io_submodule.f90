submodule(mod_io) mod_io_submodule

  implicit none

contains

  module subroutine read_binary_file_1d(filename, dtype, nrec, array)
    character(len=*), intent(in) :: filename
    integer(ik), intent(in) :: dtype, nrec
    real(rk), allocatable, intent(in out) :: array(:)
    integer(ik) :: fileunit
    allocate(array(nrec))
    open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * nrec, status='old')
    read(fileunit, rec=1) array
    close(fileunit)
  end subroutine read_binary_file_1d

  module subroutine read_binary_file_2d(filename, dtype, dsize, nrec, array)
    character(len=*), intent(in) :: filename
    integer(ik), intent(in) :: dtype, dsize, nrec
    real(rk), allocatable, intent(in out) :: array(:,:)
    integer(ik) :: fileunit, i
    allocate(array(dsize, nrec))
    open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * dsize, status='old')
    do i = 1, nrec
      read(fileunit, rec=i) array(:,i)
    end do
    close(fileunit)
  end subroutine read_binary_file_2d

end submodule mod_io_submodule
