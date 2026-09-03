program test_input4d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: input, layer

  implicit none

  type(layer) :: test_layer
  real, allocatable :: output(:,:,:,:)
  logical :: ok = .true.

  test_layer = input(3, 8, 8, 8)

  if (.not. test_layer % name == 'input') then
    ok = .false.
    write(stderr, '(a)') 'input4d layer has its name set correctly.. failed'
  end if

  if (.not. test_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'input4d layer should be marked as initialized.. failed'
  end if

  if (.not. all(test_layer % layer_shape == [3, 8, 8, 8])) then
    ok = .false.
    write(stderr, '(a)') 'input4d layer is created with requested size.. failed'
  end if

  if (.not. size(test_layer % input_layer_shape) == 0) then
    ok = .false.
    write(stderr, '(a)') 'input4d layer has no input layer shape.. failed'
  end if

  call test_layer % get_output(output)

  if (.not. all(output == 0)) then
    ok = .false.
    write(stderr, '(a)') 'input4d layer values are all initialized to 0.. failed'
  end if

  if (ok) then
    print '(a)', 'test_input4d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_input4d_layer: One or more tests failed.'
    stop 1
  end if

end program test_input4d_layer
