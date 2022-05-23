program test_flatten_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dense, flatten, input, layer, network
  use nf_flatten_layer, only: flatten_layer
  use nf_input3d_layer, only: input3d_layer

  implicit none

  type(layer) :: test_layer, input_layer
  type(network) :: net
  real, allocatable :: input_data(:,:,:), gradient(:,:,:)
  real, allocatable :: output(:)
  logical :: ok = .true.

  test_layer = flatten()

  if (.not. test_layer % name == 'flatten') then
    ok = .false.
    write(stderr, '(a)') 'flatten layer has its name set correctly.. failed'
  end if

  if (test_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer is not initialized yet.. failed'
  end if

  input_layer = input([1, 2, 2])
  call test_layer % init(input_layer)

  if (.not. test_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer is now initialized.. failed'
  end if

  if (.not. all(test_layer % layer_shape == [4])) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer has an incorrect output shape.. failed'
  end if

  ! Test forward pass - reshaping from 3-d to 1-d

  select type(this_layer => input_layer % p); type is(input3d_layer)
    call this_layer % set(reshape(real([1, 2, 3, 4]), [1, 2, 2]))
  end select

  call test_layer % forward(input_layer)
  call test_layer % get_output(output)

  if (.not. all(output == [1, 2, 3, 4])) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer correctly propagates forward.. failed'
  end if

  ! Test backward pass - reshaping from 1-d to 3-d

  ! Calling backward() will set the values on the gradient component
  ! input_layer is used only to determine shape
  call test_layer % backward(input_layer, real([1, 2, 3, 4]))

  select type(this_layer => test_layer % p); type is(flatten_layer)
    gradient = this_layer % gradient
  end select

  if (.not. all(gradient == reshape(real([1, 2, 3, 4]), [1, 2, 2]))) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer correctly propagates backward.. failed'
  end if

  net = network([ &
    input([1, 28, 28]), &
    flatten(), &
    dense(10) &
  ])

  ! Test that the output layer receives 784 elements in the input
  if (.not. all(net % layers(3) % input_layer_shape == [784])) then
    ok = .false.
    write(stderr, '(a)') 'flatten layer correctly chains input3d to dense.. failed'
  end if

  if (ok) then
    print '(a)', 'test_flatten_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_flatten_layer: One or more tests failed.'
    stop 1
  end if

end program test_flatten_layer
