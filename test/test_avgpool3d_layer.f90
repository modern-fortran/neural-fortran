program test_avgpool3d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: avgpool, input, layer
  use nf_input4d_layer, only: input4d_layer
  use nf_avgpool3d_layer, only: avgpool3d_layer

  implicit none

  type(layer) :: avgpool_layer, input_layer
  integer, parameter :: pool_size = 2, stride = 2
  integer, parameter :: channels = 3, width = 8
  integer, parameter :: input_shape(4) = [channels, width, width, width]
  integer, parameter :: output_shape(4) = [channels, width / 2, width / 2, width / 2]
  real, allocatable :: sample_input(:,:,:,:), output(:,:,:,:), gradient(:,:,:,:)
  integer :: i, j, k, ii, jj, kk
  real :: expected
  logical :: ok = .true., gradient_ok = .true.

  avgpool_layer = avgpool(pool_size, pool_size, pool_size, stride)

  if (.not. avgpool_layer % name == 'avgpool3d') then
    ok = .false.
    write(stderr, '(a)') 'avgpool3d layer has its name set correctly.. failed'
  end if

  if (avgpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'avgpool3d layer should not be marked as initialized yet.. failed'
  end if

  input_layer = input(channels, width, width, width)
  call avgpool_layer % init(input_layer)

  if (.not. avgpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'avgpool3d layer should now be marked as initialized.. failed'
  end if

  if (.not. all(avgpool_layer % input_layer_shape == input_shape)) then
    ok = .false.
    write(stderr, '(a)') 'avgpool3d layer input layer shape should be correct.. failed'
  end if

  if (.not. all(avgpool_layer % layer_shape == output_shape)) then
    ok = .false.
    write(stderr, '(a)') 'avgpool3d layer output layer shape should be correct.. failed'
  end if

  allocate(sample_input(channels, width, width, width))
  do concurrent(i = 1:width, j = 1:width, k = 1:width)
    sample_input(:,i,j,k) = i * j * k
  end do

  select type(this_layer => input_layer % p); type is(input4d_layer)
    call this_layer % set(sample_input)
  end select

  call avgpool_layer % forward(input_layer)
  call avgpool_layer % get_output(output)

  do k = 1, width / 2
    do j = 1, width / 2
      do i = 1, width / 2
        expected = 0
        do kk = 0, 1
          do jj = 0, 1
            do ii = 0, 1
              expected = expected + real((2*i-1+ii)*(2*j-1+jj)*(2*k-1+kk))
            end do
          end do
        end do
        expected = expected / 8.0
        if (.not. all(abs(output(:,i,j,k) - expected) < 1e-6)) then
          ok = .false.
          write(stderr, '(a)') 'avgpool3d layer forward pass correctly propagates the avg value.. failed'
        end if
      end do
    end do
  end do

  allocate(gradient, source=output)
  call avgpool_layer % backward(input_layer, gradient)

  select type(this_layer => avgpool_layer % p); type is(avgpool3d_layer)
    do k = 1, width
      do j = 1, width
        do i = 1, width
          if (.not. all(abs(this_layer % gradient(:,i,j,k) - &
            gradient(:,(i+1)/2,(j+1)/2,(k+1)/2) / 8.0) < 1e-6)) gradient_ok = .false.
        end do
      end do
    end do
  end select

  if (.not. gradient_ok) then
    ok = .false.
    write(stderr, '(a)') 'avgpool3d layer backward pass produces the correct dL/dx.. failed'
  end if

  if (ok) then
    print '(a)', 'test_avgpool3d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_avgpool3d_layer: One or more tests failed.'
    stop 1
  end if

end program test_avgpool3d_layer
