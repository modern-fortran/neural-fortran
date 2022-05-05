program test_conv2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv2d, input, layer
  implicit none
  type(layer) :: conv_layer, input_layer
  integer, parameter :: window_size = 3, filters = 32
  logical :: ok = .true.

  conv_layer = conv2d(window_size, filters)

  if (.not. conv_layer % name == 'conv2d') then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer has its name set correctly.. failed'
  end if

  if (conv_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer should not be marked as initialized yet.. failed'
  end if

  if (.not. conv_layer % activation == 'sigmoid') then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer is defaults to sigmoid activation.. failed'
  end if

  input_layer = input([28, 28, 3])
  call conv_layer % init(input_layer)

  if (.not. conv_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer should now be marked as initialized.. failed'
  end if

  if (.not. all(conv_layer % input_layer_shape == [28, 28, 3])) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer input layer shape should be correct.. failed'
  end if

  if (.not. all(conv_layer % layer_shape == [26, 26, filters])) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer input layer shape should be correct.. failed'
  end if

  if (ok) then
    print '(a)', 'test_conv2d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_conv2d_layer: One or more tests failed.'
    stop 1
  end if

end program test_conv2d_layer
