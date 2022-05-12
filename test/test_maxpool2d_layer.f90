program test_maxpool2d_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: maxpool2d, input, layer
  use nf_input3d_layer, only: input3d_layer

  implicit none

  type(layer) :: maxpool_layer, input_layer
  integer, parameter :: pool_size = 2, stride = 2
  real, allocatable :: sample_input(:,:,:), output(:,:,:)
  logical :: ok = .true.

  maxpool_layer = maxpool2d(pool_size)

  if (.not. maxpool_layer % name == 'maxpool2d') then
    ok = .false.
    write(stderr, '(a)') 'maxpool2d layer has its name set correctly.. failed'
  end if

  if (maxpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'maxpool2d layer should not be marked as initialized yet.. failed'
  end if

  input_layer = input([3, 32, 32])
  call maxpool_layer % init(input_layer)

  if (.not. maxpool_layer % initialized) then
    ok = .false.
    write(stderr, '(a)') 'maxpool2d layer should now be marked as initialized.. failed'
  end if

  if (.not. all(maxpool_layer % input_layer_shape == [3, 32, 32])) then
    ok = .false.
    write(stderr, '(a)') 'maxpool2d layer input layer shape should be correct.. failed'
  end if

  if (.not. all(maxpool_layer % layer_shape == [3, 16, 16])) then
    ok = .false.
    write(stderr, '(a)') 'maxpool2d layer input layer shape should be correct.. failed'
  end if

  if (ok) then
    print '(a)', 'test_maxpool2d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_maxpool2d_layer: One or more tests failed.'
    stop 1
  end if

end program test_maxpool2d_layer
