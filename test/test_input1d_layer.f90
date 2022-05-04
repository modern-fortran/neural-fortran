program test_input1d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: input, layer
  use nf_input1d_layer, only: input1d_layer
  implicit none
  type(layer) :: test_layer
  real, allocatable :: output(:)
  logical :: ok = .true.

  test_layer = input(3)

  if (.not. test_layer % name == 'input') then
    ok = .false.
    write(stderr, '(a)') 'input1d layer has its name set correctly.. failed'
  end if

  if (.not. test_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'input1d layer should be marked as initialized.. failed'
  end if

  if (.not. all(test_layer % layer_shape == [3])) then
    ok = .false.
    write(stderr, '(a)') 'input1d layer is created with requested size.. failed'
  end if

  if (.not. size(test_layer % input_layer_shape) == 0) then
    ok = .false.
    write(stderr, '(a)') 'input1d layer has no input layer shape.. failed'
  end if

  call test_layer % get_output(output) 

  if (.not. all(output == 0)) then
    ok = .false.
    write(stderr, '(a)') 'input1d layer values are all initialized to 0.. failed'
  end if

  select type(input_layer => test_layer % p); type is(input1d_layer)
    call input_layer % set([1., 2., 3.])
  end select

  call test_layer % get_output(output) 

  if (.not. all(output == [1., 2., 3.])) then
    ok = .false.
    write(stderr, '(a)') 'input1d layer can have its values set.. failed'
  end if

  if (ok) then
    print '(a)', 'test_input1d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_input1d_layer: One or more tests failed.'
    stop 1
  end if

end program test_input1d_layer
