submodule(nf_datasets) nf_datasets_submodule

  implicit none

  integer, parameter :: message_len = 128

contains

  module subroutine download_and_unpack(url)
    character(*), intent(in) :: url
    character(:), allocatable :: command, error_message, filename
    integer :: cmdstat, exitstat
    character(message_len) :: cmdmsg

    filename = url(index(url, '/', back=.true.) + 1:)
    command = 'curl -LO ' // url

    call execute_command_line(command, wait=.true., &
      exitstat=exitstat, cmdstat=cmdstat, cmdmsg=cmdmsg)

    if (any([exitstat, cmdstat] /= 0)) then
      error_message = 'cmd "' // command // '" failed'
      if (cmdstat /= 0) &
        error_message = error_message // " with message " // trim(cmdmsg)
      error stop error_message
    end if
    
    command = 'tar xvzf ' // filename

    call execute_command_line(command, wait=.true., &
      exitstat=exitstat, cmdstat=cmdstat, cmdmsg=cmdmsg)

    if (any([exitstat, cmdstat] /= 0)) then
      error_message = 'cmd "' // command // '" failed'
      if (cmdstat /= 0) &
        error_message = error_message // " with message " // trim(cmdmsg)
      error stop  error_message
    end if

  end subroutine download_and_unpack

end submodule nf_datasets_submodule
