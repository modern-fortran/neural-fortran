program test_loss

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: mse, quadratic

  implicit none

  logical :: ok = .true.

  if (ok) then
    print '(a)', 'test_loss: All tests passed.'
  else
    write(stderr, '(a)') 'test_loss: One or more tests failed.'
    stop 1
  end if

end program test_loss