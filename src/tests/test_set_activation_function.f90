program test_set_activation_function

  ! This program will test whether per-network and per-layer
  ! setting of activation functions works as expected.
  ! First we create an array of random variables.
  ! Then we set different activation functions to different 
  ! layers in the network. 
  ! Finally, we test whether each function produces same 
  ! values as the activation functions set in the layers. 

  use mod_activation
  use mod_network, only: network_type
  use mod_random, only: randn

  implicit none
  type(network_type) :: net
  real, allocatable :: x(:)
  integer :: n
  logical, allocatable :: tests(:)
  
  tests = [logical ::]

  x = randn(100)

  ! the network will be created with 
  ! sigmoid activation functions for all layers
  net = network_type([1, 1, 1, 1, 1])

  do n = 1, size(net % layers)
    tests = [tests, all(sigmoid(x) == net % layers(n) % activation(x))]
    tests = [tests, all(sigmoid_prime(x) == net % layers(n) % activation_prime(x))]
  end do

  ! now set the various functions for other layers
  call net % layers(2) % set_activation('gaussian')
  call net % layers(3) % set_activation('step')
  call net % layers(4) % set_activation('tanh')
  call net % layers(5) % set_activation('relu')
    
  tests = [tests, all(sigmoid(x) == net % layers(1) % activation(x))]
  tests = [tests, all(sigmoid_prime(x) == net % layers(1) % activation_prime(x))]

  tests = [tests, all(gaussian(x) == net % layers(2) % activation(x))]
  tests = [tests, all(gaussian_prime(x) == net % layers(2) % activation_prime(x))]

  tests = [tests, all(step(x) == net % layers(3) % activation(x))]
  tests = [tests, all(step_prime(x) == net % layers(3) % activation_prime(x))]

  tests = [tests, all(tanhf(x) == net % layers(4) % activation(x))]
  tests = [tests, all(tanh_prime(x) == net % layers(4) % activation_prime(x))]

  tests = [tests, all(relu(x) == net % layers(5) % activation(x))]
  tests = [tests, all(relu_prime(x) == net % layers(5) % activation_prime(x))]

  print *, tests

  if (all(tests)) then
    print *, 'All tests passed.'
  else
    error stop 'some tests failed.'
  end if

end program test_set_activation_function
