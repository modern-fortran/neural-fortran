program test_dense_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dense, layer
  implicit none
  type(layer) :: layer1, layer2
  logical :: ok = .true.

  layer1 = dense(10)

  if (.not. layer1 % name == 'dense') then
    ok = .false.
    write(stderr, '(a)') 'dense layer has its name set correctly.. failed'
  end if

  if (.not. all(layer1 % layer_shape == [10])) then
    ok = .false.
    write(stderr, '(a)') 'dense layer is created with requested size.. failed'
  end if

  if (layer1 % initialized) then
    ok = .false.
    write(stderr, '(a)') 'dense layer should not be marked as initialized yet.. failed'
  end if

  if (.not. layer1 % activation == 'sigmoid') then
    ok = .false.
    write(stderr, '(a)') 'dense layer is defaults to sigmoid activation.. failed'
  end if

  layer1 = dense(10, activation='relu')

  if (.not. layer1 % activation == 'relu') then
    ok = .false.
    write(stderr, '(a)') 'dense layer is created with the specified activation.. failed'
  end if

  layer2 = dense(20)
  call layer2 % init(layer1)

  if (.not. layer2 % initialized) then
    ok = .false.
    write(stderr, '(a)') 'dense layer should now be marked as initialized.. failed'
  end if

  if (.not. all(layer2 % input_layer_shape == [10])) then
    ok = .false.
    write(stderr, '(a)') 'dense layer should have a correct input layer shape.. failed'
  end if

  if (ok) then
    print '(a)', 'test_dense_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_dense_layer: One or more tests failed.'
    stop 1
  end if

end program test_dense_layer
