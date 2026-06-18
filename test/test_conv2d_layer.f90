program test_conv2d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv, input, layer
  use nf_input3d_layer, only: input3d_layer

  implicit none

  type(layer) :: conv_layer, input_layer
  integer, parameter :: filters = 32, kernel_size=3
  real, allocatable :: sample_input(:,:,:), output(:,:,:)
  real, parameter :: tolerance = 1e-7
  logical :: ok = .true.

  conv_layer = conv(filters, kernel_size, kernel_size)

  if (.not. conv_layer % name == 'conv2d') then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer has its name set correctly.. failed'
  end if

  if (conv_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer should not be marked as initialized yet.. failed'
  end if

  if (.not. conv_layer % activation == 'relu') then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer defaults to relu activation.. failed'
  end if

  input_layer = input(3, 32, 32)
  call conv_layer % init(input_layer)

  if (.not. conv_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer should now be marked as initialized.. failed'
  end if

  if (.not. all(conv_layer % input_layer_shape == [3, 32, 32])) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer input layer shape should be correct.. failed'
  end if

  if (.not. all(conv_layer % layer_shape == [filters, 30, 30])) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer input layer shape should be correct.. failed'
  end if

  ! Minimal conv2d layer: 1 channel, 3x3 pixel image;
  allocate(sample_input(1, 3, 3))
  sample_input = 0

  input_layer = input(1, 3, 3)
  conv_layer = conv(filters, kernel_size, kernel_size)
  call conv_layer % init(input_layer)

  select type(this_layer => input_layer % p); type is(input3d_layer)
    call this_layer % set(sample_input)
  end select

  deallocate(sample_input)

  call conv_layer % forward(input_layer)
  call conv_layer % get_output(output)

  if (.not. all(abs(output) < tolerance)) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer with zero input and sigmoid function must forward to all 0.5.. failed'
  end if

  ! Minimal conv2d layer: 1 channel, 17x17 pixel image, stride=3;
  allocate(sample_input(1, 17, 17))
  sample_input = 0

  input_layer = input(1, 17, 17)
  conv_layer = conv(filters, kernel_size, kernel_size, stride=[3, 4])
  call conv_layer % init(input_layer)

  select type(this_layer => input_layer % p); type is(input3d_layer)
    call this_layer % set(sample_input)
  end select

  deallocate(sample_input)

  call conv_layer % forward(input_layer)
  call conv_layer % get_output(output)

  if (.not. all(abs(output) < tolerance)) then
    ok = .false.
    write(stderr, '(a)') 'conv2d layer with zero input and sigmoid function must forward to all 0.5.. failed'
  end if

  ! Summary
  if (ok) then
    print '(a)', 'test_conv2d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_conv2d_layer: One or more tests failed.'
    stop 1
  end if

end program test_conv2d_layer
