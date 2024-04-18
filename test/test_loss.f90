program test_loss

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: mse, quadratic

  implicit none

  logical :: ok = .true.

  block

    type(mse) :: loss
    real :: true(2) = [1., 2.]
    real :: pred(2) = [3., 4.]

    if (.not. loss % eval(true, pred) == 4) then
      write(stderr, '(a)') 'expected output of mse % eval().. failed'
      ok = .false.
    end if

    if (.not. all(loss % derivative(true, pred) == [2, 2])) then
      write(stderr, '(a)') 'expected output of mse % derivative().. failed'
      ok = .false.
    end if

  end block

  block

    type(quadratic) :: loss
    real :: true(4) = [1., 2., 3., 4.]
    real :: pred(4) = [3., 4., 5., 6.]

    if (.not. loss % eval(true, pred) == 8) then
      write(stderr, '(a)') 'expected output of quadratic % eval().. failed'
      ok = .false.
    end if

    if (.not. all(loss % derivative(true, pred) == [2, 2, 2, 2])) then
      write(stderr, '(a)') 'expected output of quadratic % derivative().. failed'
      ok = .false.
    end if

  end block

  if (ok) then
    print '(a)', 'test_loss: All tests passed.'
  else
    write(stderr, '(a)') 'test_loss: One or more tests failed.'
    stop 1
  end if

end program test_loss