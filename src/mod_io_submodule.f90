submodule(mod_io) mod_io_submodule

  implicit none

  integer, parameter :: message_len = 128
contains

  subroutine download_and_uncompress()
    character(len=*), parameter :: download_mechanism = 'curl -LO '
    character(len=*), parameter :: base_url='https://github.com/modern-fortran/neural-fortran/files/8498876/'
    character(len=*), parameter :: download_filename = 'mnist.tar.gz'
    character(len=*), parameter :: download_command = download_mechanism // base_url //download_filename
    character(len=*), parameter :: uncompress_file= 'tar xvzf ' // download_filename
    character(len=message_len) :: command_message
    character(len=:), allocatable :: error_message
    integer exit_status, command_status
    exit_status=0
    call execute_command_line(command=download_command, &
      wait=.true., exitstat=exit_status, cmdstat=command_status, cmdmsg=command_message)
    if (any([exit_status, command_status]/=0)) then
      error_message = 'command "' // download_command // '" failed'
      if (command_status/=0) error_message = error_message // " with message " // trim(command_message)
      error stop  error_message
    end if
    call execute_command_line(command=uncompress_file , &
      wait=.true., exitstat=exit_status, cmdstat=command_status, cmdmsg=command_message)
    if (any([exit_status, command_status]/=0)) then
      error_message = 'command "' // uncompress_file // '" failed'
      if (command_status/=0) error_message = error_message // " with message " // trim(command_message)
      error stop  error_message
    end if
  end subroutine

  module subroutine read_binary_file_1d(filename, dtype, nrec, array)
    character(len=*), intent(in) :: filename
    integer(ik), intent(in) :: dtype, nrec
    real(rk), allocatable, intent(in out) :: array(:)
    integer(ik) :: fileunit
    character(len=message_len) io_message, command_message
    integer io_status
    io_status=0
    open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * nrec, status='old', iostat=io_status)
    if (io_status/=0) then
      call download_and_uncompress
      open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * nrec, status='old', iostat=io_status, iomsg=io_message)
      if (io_status/=0) error stop trim(io_message)
    end if
    allocate(array(nrec))
    read(fileunit, rec=1) array
    close(fileunit)
  end subroutine read_binary_file_1d

  module subroutine read_binary_file_2d(filename, dtype, dsize, nrec, array)
    character(len=*), intent(in) :: filename
    integer(ik), intent(in) :: dtype, dsize, nrec
    real(rk), allocatable, intent(in out) :: array(:,:)
    integer(ik) :: fileunit, i
    character(len=message_len) io_message, command_message
    integer io_status
    open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * nrec, status='old', iostat=io_status)
    if (io_status/=0) then
      call download_and_uncompress
      open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * nrec, status='old', iostat=io_status, iomsg=io_message)
      if (io_status/=0) error stop trim(io_message)
    end if
    allocate(array(dsize, nrec))
    open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * dsize, status='old')
    do i = 1, nrec
      read(fileunit, rec=i) array(:,i)
    end do
    close(fileunit)
  end subroutine read_binary_file_2d

end submodule mod_io_submodule
