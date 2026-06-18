program test_avgpool1d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: avgpool, input, layer
  use nf_input2d_layer, only: input2d_layer
  use nf_avgpool1d_layer, only: avgpool1d_layer

  implicit none

  type(layer) :: avgpool_layer, input_layer
  integer, parameter :: pool_size = 2, stride = 2
  integer, parameter :: channels = 3, length = 32
  integer, parameter :: input_shape(2) = [channels, length]
  integer, parameter :: output_shape(2) = [channels, length / 2]
  real, allocatable :: sample_input(:,:), output(:,:), gradient(:,:)
  integer :: i
  logical :: ok = .true., gradient_ok = .true.

  avgpool_layer = avgpool(pool_size, stride)

  if (.not. avgpool_layer % name == 'avgpool1d') then
    ok = .false.
    write(stderr, '(a)') 'avgpool1d layer has its name set correctly.. failed'
  end if

  if (avgpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'avgpool1d layer should not be marked as initialized yet.. failed'
  end if

  input_layer = input(channels, length)
  call avgpool_layer % init(input_layer)

  if (.not. avgpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'avgpool1d layer should now be marked as initialized.. failed'
  end if

  if (.not. all(avgpool_layer % input_layer_shape == input_shape)) then
    ok = .false.
    write(stderr, '(a)') 'avgpool1d layer input layer shape should be correct.. failed'
  end if

  if (.not. all(avgpool_layer % layer_shape == output_shape)) then
    ok = .false.
    write(stderr, '(a)') 'avgpool1d layer output layer shape should be correct.. failed'
  end if

  ! Allocate and initialize sample input data
  allocate(sample_input(channels, length))
  do concurrent(i = 1:length)
    sample_input(:,i) = i
  end do

  select type(this_layer => input_layer % p); type is(input2d_layer)
    call this_layer % set(sample_input)
  end select

  call avgpool_layer % forward(input_layer)
  call avgpool_layer % get_output(output)

  do i = 1, length / 2
    ! For input values [1,2,3,4,...], avgpool1d with pool_size=2, stride=2:
    ! output(:,i) = avg of [2*i-1, 2*i] = (2*i-1 + 2*i)/2 = (4*i-1)/2
    if (.not. all(output(:,i) == (4*i-1)/2.0)) then
      ok = .false.
      write(stderr, '(a)') 'avgpool1d layer forward pass correctly propagates the avg value.. failed'
    end if
  end do

  ! Test the backward pass
  allocate(gradient, source=output)
  call avgpool_layer % backward(input_layer, gradient)

  select type(this_layer => avgpool_layer % p); type is(avgpool1d_layer)
    do i = 1, length
      ! For avgpool1d, each input in a pool window receives gradient(:,i/2) / pool_size if i is even,
      ! and gradient(:,(i+1)/2) / pool_size if i is odd (since stride=2, pool_size=2)
      if (mod(i,2) == 0) then
        if (.not. all(this_layer % gradient(:,i) == gradient(:,i/2) / 2.0)) gradient_ok = .false.
      else
        if (.not. all(this_layer % gradient(:,i) == gradient(:,(i+1)/2) / 2.0)) gradient_ok = .false.
      end if
    end do
  end select

  if (.not. gradient_ok) then
    ok = .false.
    write(stderr, '(a)') 'avgpool1d layer backward pass produces the correct dL/dx.. failed'
  end if

  if (ok) then
    print '(a)', 'test_avgpool1d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_avgpool1d_layer: One or more tests failed.'
    stop 1
  end if

end program test_avgpool1d_layer
