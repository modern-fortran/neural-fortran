program test_flatten_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: flatten, layer
  use nf_flatten_layer, only: flatten_layer

  implicit none

  type(layer) :: test_layer
  real, allocatable :: output(:)
  logical :: ok = .true.

  test_layer = flatten()

  if (.not. test_layer % name == 'flatten') then
    ok = .false.
    write(stderr, '(a)') 'flatten layer has its name set correctly.. failed'
  end if

  if (test_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer is not initialized yet.. failed'
  end if

  if (ok) then
    print '(a)', 'test_flatten_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_flatten_layer: One or more tests failed.'
    stop 1
  end if

end program test_flatten_layer
