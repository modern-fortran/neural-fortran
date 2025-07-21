program test_avgpool2d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: avgpool2d, input, layer
  use nf_input3d_layer, only: input3d_layer
  use nf_avgpool2d_layer, only: avgpool2d_layer

  implicit none

  type(layer) :: avgpool_layer, input_layer
  integer, parameter :: pool_size = 2, stride = 2
  integer, parameter :: channels = 3, width = 32
  integer, parameter :: input_shape(3) = [channels, width, width]
  integer, parameter :: output_shape(3) = [channels, width / 2, width / 2]
  real, allocatable :: sample_input(:,:,:), output(:,:,:), gradient(:,:,:)
  integer :: i, j
  logical :: ok = .true., gradient_ok = .true.

  avgpool_layer = avgpool2d(pool_size)

  if (.not. avgpool_layer % name == 'avgpool2d') then
    ok = .false.
    write(stderr, '(a)') 'avgpool2d layer has its name set correctly.. failed'
  end if

  if (avgpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'avgpool2d layer should not be marked as initialized yet.. failed'
  end if

  input_layer = input(channels, width, width)
  call avgpool_layer % init(input_layer)

  if (.not. avgpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'avgpool2d layer should now be marked as initialized.. failed'
  end if

  if (.not. all(avgpool_layer % input_layer_shape == input_shape)) then
    ok = .false.
    write(stderr, '(a)') 'avgpool2d layer input layer shape should be correct.. failed'
  end if

  if (.not. all(avgpool_layer % layer_shape == output_shape)) then
    ok = .false.
    write(stderr, '(a)') 'avgpool2d layer input layer shape should be correct.. failed'
  end if

  ! Allocate and initialize sample input data
  allocate(sample_input(channels, width, width))
  do concurrent(i = 1:width, j = 1:width)
    sample_input(:,i,j) = i * j
  end do

  select type(this_layer => input_layer % p); type is(input3d_layer)
    call this_layer % set(sample_input)
  end select

  call avgpool_layer % forward(input_layer)
  call avgpool_layer % get_output(output)

  do j = 1, width / 2
    do i = 1, width / 2
      ! For input(:,i,j) = i*j, avgpool2d with pool_size=2, stride=2:
      ! window: (2*i-1,2*j-1), (2*i,2*j-1), (2*i-1,2*j), (2*i,2*j)
      ! avg = (a + b + c + d) / 4
      if (.not. all(output(:,i,j) == ( &
            (2*i-1)*(2*j-1) + (2*i)*(2*j-1) + (2*i-1)*(2*j) + (2*i)*(2*j) &
            ) / 4.0)) then
        ok = .false.
        write(stderr, '(a)') 'avgpool2d layer forward pass correctly propagates the avg value.. failed'
      end if
    end do
  end do

  ! Test the backward pass
  ! Allocate and initialize the downstream gradient field
  allocate(gradient, source=output)

  ! Make a backward pass
  call avgpool_layer % backward(input_layer, gradient)

  select type(this_layer => avgpool_layer % p); type is(avgpool2d_layer)
    do j = 1, width
      do i = 1, width
        ! Each input in a 2x2 window receives gradient(:,i/2,j/2) / 4 if both i,j even,
        ! gradient(:,(i+1)/2,(j+1)/2) / 4 if both i,j odd, etc.
        if (mod(i,2) == 0 .and. mod(j,2) == 0) then
          if (.not. all(this_layer % gradient(:,i,j) == gradient(:,i/2,j/2) / 4.0)) gradient_ok = .false.
        else if (mod(i,2) == 1 .and. mod(j,2) == 0) then
          if (.not. all(this_layer % gradient(:,i,j) == gradient(:,(i+1)/2,j/2) / 4.0)) gradient_ok = .false.
        else if (mod(i,2) == 0 .and. mod(j,2) == 1) then
          if (.not. all(this_layer % gradient(:,i,j) == gradient(:,i/2,(j+1)/2) / 4.0)) gradient_ok = .false.
        else
          if (.not. all(this_layer % gradient(:,i,j) == gradient(:,(i+1)/2,(j+1)/2) / 4.0)) gradient_ok = .false.
        end if
      end do
    end do
  end select

  if (.not. gradient_ok) then
    ok = .false.
    write(stderr, '(a)') 'avgpool2d layer backward pass produces the correct dL/dx.. failed'
  end if

  if (ok) then
    print '(a)', 'test_avgpool2d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_avgpool2d_layer: One or more tests failed.'
    stop 1
  end if

end program test_avgpool2d_layer
