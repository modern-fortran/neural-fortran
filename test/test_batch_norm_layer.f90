program test_batch_norm_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: batch_norm, input, layer
  use nf_input3d_layer, only: input3d_layer
  use nf_batch_norm_layer, only: batch_norm_layer

  implicit none

  type(layer) :: bn_layer, input_layer
  integer, parameter :: num_features = 64
  real, allocatable :: sample_input(:,:)
  real, allocatable :: output(:,:)
  real, allocatable :: gradient(:,:)
  integer, parameter :: input_shape(1) = [num_features]
  real, allocatable :: gamma_grad(:), beta_grad(:)
  real, parameter :: tolerance = 1e-7
  logical :: ok = .true.

  bn_layer = batch_norm(num_features)

  if (.not. bn_layer % name == 'batch_norm') then
    ok = .false.
    write(stderr, '(a)') 'batch_norm layer has its name set correctly.. failed'
  end if

  if (bn_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'batch_norm layer should not be marked as initialized yet.. failed'
  end if

  input_layer = input(input_shape)
  call bn_layer % init(input_layer)

  if (.not. bn_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'batch_norm layer should now be marked as initialized.. failed'
  end if

  if (.not. all(bn_layer % input_layer_shape == [num_features])) then
    ok = .false.
    write(stderr, '(a)') 'batch_norm layer input layer shape should be correct.. failed'
  end if

  ! Initialize sample input and gradient
  allocate(sample_input(num_features, 1))
  allocate(gradient(num_features, 1))
  sample_input = 1.0
  gradient = 2.0

  ! Set input for the input layer
  select type(this_layer => input_layer % p); type is(input3d_layer)
    call this_layer % set(sample_input)
  end select

  ! Initialize the batch normalization layer
  bn_layer = batch_norm(num_features)
  call bn_layer % init(input_layer)

  ! Perform forward and backward passes
  call bn_layer % forward(input_layer)
  call bn_layer % backward(input_layer, gradient)

  ! Retrieve output and check normalization
  call bn_layer % get_output(output)
  if (.not. all(abs(output - sample_input) < tolerance)) then
    ok = .false.
    write(stderr, '(a)') 'batch_norm layer output should be close to input.. failed'
  end if

  ! Retrieve gamma and beta gradients
  allocate(gamma_grad(num_features))
  allocate(beta_grad(num_features))
  call bn_layer % get_gradients(gamma_grad, beta_grad)

  if (.not. all(beta_grad == sum(gradient))) then
    ok = .false.
    write(stderr, '(a)') 'batch_norm layer beta gradients are incorrect.. failed'
  end if

  ! Report test results
  if (ok) then
    print '(a)', 'test_batch_norm_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_batch_norm_layer: One or more tests failed.'
    stop 1
  end if

end program test_batch_norm_layer
