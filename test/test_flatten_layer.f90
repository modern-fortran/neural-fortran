program test_flatten_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: flatten, input, layer
  use nf_flatten_layer, only: flatten_layer
  use nf_input3d_layer, only: input3d_layer

  implicit none

  type(layer) :: test_layer, input_layer
  real, allocatable :: input_data(:,:,:)
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

  input_layer = input([1, 2, 2])
  call test_layer % init(input_layer)

  if (.not. test_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer is now initialized.. failed'
  end if

  if (.not. all(test_layer % layer_shape == [4])) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer has an incorrect output shape.. failed'
  end if

  select type(this_layer => input_layer % p); type is(input3d_layer)
    call this_layer % set(reshape(real([1, 2, 3, 4]), [1, 2, 2]))
  end select

  call test_layer % forward(input_layer)
  call test_layer % get_output(output)

  if (.not. all(output == [1, 2, 3, 4])) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer correctly propagates forward.. failed'
  end if

  if (ok) then
    print '(a)', 'test_flatten_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_flatten_layer: One or more tests failed.'
    stop 1
  end if

end program test_flatten_layer
