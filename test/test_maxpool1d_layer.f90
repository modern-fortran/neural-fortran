program test_maxpool1d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: maxpool, input, layer
  use nf_input2d_layer, only: input2d_layer
  use nf_maxpool1d_layer, only: maxpool1d_layer

  implicit none

  type(layer) :: maxpool_layer, input_layer
  integer, parameter :: pool_size = 2, stride = 2
  integer, parameter :: channels = 3, length = 32
  integer, parameter :: input_shape(2) = [channels, length]
  integer, parameter :: output_shape(2) = [channels, length / 2]
  real, allocatable :: sample_input(:,:), output(:,:), gradient(:,:)
  integer :: i
  logical :: ok = .true., gradient_ok = .true.

  maxpool_layer = maxpool(pool_width=pool_size, stride=stride)

  if (.not. maxpool_layer % name == 'maxpool1d') then
    ok = .false.
    write(stderr, '(a)') 'maxpool1d layer has its name set correctly.. failed'
  end if

  if (maxpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'maxpool1d layer should not be marked as initialized yet.. failed'
  end if

  input_layer = input(channels, length)
  call maxpool_layer % init(input_layer)

  if (.not. maxpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'maxpool1d layer should now be marked as initialized.. failed'
  end if

  if (.not. all(maxpool_layer % input_layer_shape == input_shape)) then
    ok = .false.
    write(stderr, '(a)') 'maxpool1d layer input layer shape should be correct.. failed'
  end if

  if (.not. all(maxpool_layer % layer_shape == output_shape)) then
    ok = .false.
    write(stderr, '(a)') 'maxpool1d layer output layer shape should be correct.. failed'
  end if

  ! Allocate and initialize sample input data
  allocate(sample_input(channels, length))
  do concurrent(i = 1:length)
    sample_input(:,i) = i
  end do

  select type(this_layer => input_layer % p); type is(input2d_layer)
    call this_layer % set(sample_input)
  end select

  call maxpool_layer % forward(input_layer)
  call maxpool_layer % get_output(output)

  do i = 1, length / 2
    if (.not. all(output(:,i) == stride * i)) then
      ok = .false.
      write(stderr, '(a)') 'maxpool1d layer forward pass correctly propagates the max value.. failed'
    end if
  end do

  ! Test the backward pass
  allocate(gradient, source=output)
  call maxpool_layer % backward(input_layer, gradient)

  select type(this_layer => maxpool_layer % p); type is(maxpool1d_layer)
    do i = 1, length
      if (mod(i,2) == 0) then
        if (.not. all(sample_input(:,i) == this_layer % gradient(:,i))) gradient_ok = .false.
      else
        if (.not. all(this_layer % gradient(:,i) == 0)) gradient_ok = .false.
      end if
    end do
  end select

  if (.not. gradient_ok) then
    ok = .false.
    write(stderr, '(a)') 'maxpool1d layer backward pass produces the correct dL/dx.. failed'
  end if

  if (ok) then
    print '(a)', 'test_maxpool1d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_maxpool1d_layer: One or more tests failed.'
    stop 1
  end if

end program test_maxpool1d_layer
