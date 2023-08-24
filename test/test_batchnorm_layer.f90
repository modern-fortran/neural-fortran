program test_batchnorm_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: batchnorm, layer
  use nf_batchnorm_layer, only: batchnorm_layer

  implicit none

  type(layer) :: bn_layer
  integer, parameter :: num_features = 64
  real, allocatable :: sample_input(:,:)
  real, allocatable :: output(:,:)
  real, allocatable :: gradient(:,:)
  integer, parameter :: input_shape(1) = [num_features]
  real, allocatable :: gamma_grad(:), beta_grad(:)
  real, parameter :: tolerance = 1e-7
  logical :: ok = .true.

  bn_layer = batchnorm(num_features)

  if (.not. bn_layer % name == 'batchnorm') then
    ok = .false.
    write(stderr, '(a)') 'batchnorm layer has its name set correctly.. failed'
  end if

  if (bn_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'batchnorm layer should not be marked as initialized yet.. failed'
  end if

  ! Initialize sample input and gradient
  allocate(sample_input(num_features, 1))
  allocate(gradient(num_features, 1))
  sample_input = 1.0
  gradient = 2.0

  !TODO run forward and backward passes directly on the batchnorm_layer instance
  !TODO since we don't yet support tiying in with the input layer.

  !TODO Retrieve output and check normalization
  !call bn_layer % get_output(output)
  !if (.not. all(abs(output - sample_input) < tolerance)) then
  !  ok = .false.
  !  write(stderr, '(a)') 'batchnorm layer output should be close to input.. failed'
  !end if

  !TODO Retrieve gamma and beta gradients
  !allocate(gamma_grad(num_features))
  !allocate(beta_grad(num_features))
  !call bn_layer % get_gradients(gamma_grad, beta_grad)

  !if (.not. all(beta_grad == sum(gradient))) then
  !  ok = .false.
  !  write(stderr, '(a)') 'batchnorm layer beta gradients are incorrect.. failed'
  !end if

  ! Report test results
  if (ok) then
    print '(a)', 'test_batchnorm_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_batchnorm_layer: One or more tests failed.'
    stop 1
  end if

end program test_batchnorm_layer
