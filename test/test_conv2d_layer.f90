program test_conv2d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv, input, layer
  use nf_input3d_layer, only: input3d_layer
  use nf_conv2d_layer, only: conv2d_layer
  use nf_activation, only: relu

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

  call check_pointwise_gradients()
  call check_window_gradients()

  ! Summary
  if (ok) then
    print '(a)', 'test_conv2d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_conv2d_layer: One or more tests failed.'
    stop 1
  end if

contains

 subroutine check_pointwise_gradients()
   type(conv2d_layer) :: pointwise
   real :: x(2,2,3), downstream(2,2,3)
   real :: expected_dx(2,2,3), expected_dw(2,2,1,1), expected_db(2)
   integer :: i, j, k, n

   pointwise = conv2d_layer(2, 1, relu(), [1,1])
   call pointwise % init([2,2,3])
   pointwise % kernel(:,:,1,1) = reshape([1., 2., 3., 4.], [2,2])
   pointwise % biases = 1
   x = reshape([(real(i), i=1,12)], shape(x))
   downstream = reshape([(real(i)/4, i=1,12)], shape(downstream))
   call pointwise % forward(x)
   call pointwise % backward(x, downstream)

   expected_dw = 0
   expected_dx = 0
   expected_db = 0
   do n = 1, 2
     expected_db(n) = sum(downstream(n,:,:))
     do k = 1, 2
       expected_dw(n,k,1,1) = sum(x(k,:,:) * downstream(n,:,:))
     end do
   end do
   do j = 1, 3
     do i = 1, 2
       do k = 1, 2
         expected_dx(k,i,j) = sum(pointwise % kernel(:,k,1,1) * downstream(:,i,j))
       end do
     end do
   end do
   if (any(abs(pointwise % dw - expected_dw) > tolerance) .or. &
       any(abs(pointwise % db - expected_db) > tolerance) .or. &
       any(abs(pointwise % gradient - expected_dx) > tolerance)) then
     ok = .false.
     write(stderr, '(a)') 'conv2d backward must reduce across output positions and filters.. failed'
   end if

   call pointwise % backward(x, downstream)
   if (any(abs(pointwise % dw - 2*expected_dw) > tolerance) .or. &
       any(abs(pointwise % db - 2*expected_db) > tolerance) .or. &
       any(abs(pointwise % gradient - expected_dx) > tolerance)) then
     ok = .false.
     write(stderr, '(a)') 'conv2d backward accumulates parameter gradients but resets input gradients.. failed'
   end if
 end subroutine check_pointwise_gradients

 subroutine check_window_gradients()
   type(conv2d_layer) :: convolution
   real :: x(2,4,5), downstream(2,2,3)
   real :: dx(2,4,5), dw(2,2,3,3), db(2)
   real, parameter :: step = 1./128, gradient_tolerance = 2e-4
   real :: saved, plus, minus, numerical, max_error
   integer :: i, j, k, n

   convolution = conv2d_layer(2, 3, relu(), [1,1])
   call convolution % init(shape(x))
   x = reshape([(real(i)/32, i=1,size(x))], shape(x))
   downstream = reshape([(real(i)/16, i=1,size(downstream))], shape(downstream))
   convolution % kernel = reshape([(real(i)/64, i=1,size(dw))], shape(dw))
   convolution % biases = 1
   call convolution % forward(x)
   if (any(convolution % z <= 0)) error stop 'gradient test requires positive ReLU inputs'
   call convolution % backward(x, downstream)
   dx = convolution % gradient
   dw = convolution % dw
   db = convolution % db

   ! Finite differences of sum(output * downstream), away from the ReLU kink.
   max_error = 0
   do j = 1, size(x,3)
     do i = 1, size(x,2)
       do k = 1, size(x,1)
         saved = x(k,i,j)
         x(k,i,j) = saved + step
         plus = weighted_output(convolution, x, downstream)
         x(k,i,j) = saved - step
         minus = weighted_output(convolution, x, downstream)
         x(k,i,j) = saved
         numerical = (plus - minus) / (2*step)
         max_error = max(max_error, abs(dx(k,i,j) - numerical) / (1 + abs(numerical)))
       end do
     end do
   end do
   do j = 1, 3
     do i = 1, 3
       do k = 1, 2
         do n = 1, 2
           saved = convolution % kernel(n,k,i,j)
           convolution % kernel(n,k,i,j) = saved + step
           plus = weighted_output(convolution, x, downstream)
           convolution % kernel(n,k,i,j) = saved - step
           minus = weighted_output(convolution, x, downstream)
           convolution % kernel(n,k,i,j) = saved
           numerical = (plus - minus) / (2*step)
           max_error = max(max_error, abs(dw(n,k,i,j) - numerical) / (1 + abs(numerical)))
         end do
       end do
     end do
   end do
   do n = 1, 2
     saved = convolution % biases(n)
     convolution % biases(n) = saved + step
     plus = weighted_output(convolution, x, downstream)
     convolution % biases(n) = saved - step
     minus = weighted_output(convolution, x, downstream)
     convolution % biases(n) = saved
     numerical = (plus - minus) / (2*step)
     max_error = max(max_error, abs(db(n) - numerical) / (1 + abs(numerical)))
   end do
   if (max_error > gradient_tolerance) then
     ok = .false.
     write(stderr, '(a,es12.4)') 'conv2d backward finite-difference gradient error: ', max_error
   end if
 end subroutine check_window_gradients

 real function weighted_output(convolution, x, downstream) result(loss)
   type(conv2d_layer), intent(in out) :: convolution
   real, intent(in) :: x(:,:,:), downstream(:,:,:)
   call convolution % forward(x)
   loss = sum(convolution % output * downstream)
 end function weighted_output

end program test_conv2d_layer
