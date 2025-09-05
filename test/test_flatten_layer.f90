program test_flatten_layer

  use iso_fortran_env, only: stderr => error_unit
  use nf,                only: dense, flatten, input, layer, network
  use nf_flatten_layer,  only: flatten_layer
  use nf_input2d_layer,  only: input2d_layer
  use nf_input3d_layer,  only: input3d_layer

  implicit none

  type(layer)   :: test_layer, input_layer
  type(network) :: net
  real, allocatable :: gradient_3d(:,:,:), gradient_2d(:,:)
  real, allocatable :: output(:)
  logical :: ok = .true.

  call banner('TEST FLATTEN')

  ! ---------- 3D INPUT ----------
  test_layer = flatten()

  call assert_true(trim(test_layer%name) == 'flatten',          &
       "flatten layer has its name set correctly.. failed", ok)

  call assert_true(.not. test_layer%initialized,                &
       "flatten layer is not initialized yet.. failed", ok)

  input_layer = input(1, 2, 2)
  call test_layer%init(input_layer)

  call assert_true(test_layer%initialized,                      &
       "flatten layer is now initialized.. failed", ok)

  call assert_true(all(test_layer%layer_shape == [4]),          &
       "flatten layer has an incorrect output shape.. failed", ok)

  ! Forward 3D -> 1D
  call set_input3d(input_layer, reshape(real([1,2,3,4]), [1,2,2]))
  call test_layer%forward(input_layer)
  call test_layer%get_output(output)

  call assert_true(size(output) == 4,                           &
       "flatten forward output size (3D) mismatch.. failed", ok)
  call assert_true(all(output == [1,2,3,4]),                    &
       "flatten layer correctly propagates forward.. failed", ok)

  ! Backward 1D -> 3D
  call test_layer%backward(input_layer, real([1,2,3,4]))
  call grab_flatten_gradients3d(test_layer, gradient_3d)

  call assert_true(allocated(gradient_3d),                      &
       "gradient_3d not allocated after backward.. failed", ok)
  call assert_true(all(gradient_3d == reshape(real([1,2,3,4]), [1,2,2])), &
       "flatten layer correctly propagates backward.. failed", ok)

  ! ---------- 2D INPUT ----------
  test_layer = flatten()
  input_layer = input(2, 3)
  call test_layer%init(input_layer)

  call assert_true(all(test_layer%layer_shape == [6]),          &
       "flatten layer has an incorrect output shape for 2D input.. failed", ok)

  ! Forward 2D -> 1D
  call set_input2d(input_layer, reshape(real([1,2,3,4,5,6]), [2,3]))
  call test_layer%forward(input_layer)
  call test_layer%get_output(output)

  call assert_true(size(output) == 6,                           &
       "flatten forward output size (2D) mismatch.. failed", ok)
  call assert_true(all(output == [1,2,3,4,5,6]),                &
       "flatten layer correctly propagates forward for 2D input.. failed", ok)

  ! Backward 1D -> 2D
  call test_layer%backward(input_layer, real([1,2,3,4,5,6]))
  call grab_flatten_gradients2d(test_layer, gradient_2d)

  call assert_true(allocated(gradient_2d),                      &
       "gradient_2d not allocated after backward.. failed", ok)
  call assert_true(all(gradient_2d == reshape(real([1,2,3,4,5,6]), [2,3])), &
       "flatten layer correctly propagates backward for 2D input.. failed", ok)

  ! ---------- CHAIN TO DENSE ----------
  net = network([ input(1,28,28), flatten(), dense(10) ])

  call assert_true(all(net%layers(3)%input_layer_shape == [784]), &
       "flatten layer correctly chains input3d to dense.. failed", ok)

  if (ok) then
    print '(a)', 'test_flatten_layer: All tests passed.'
  else
    write(stderr,'(a)') 'test_flatten_layer: One or more tests failed.'
    stop 1
  end if

contains

  subroutine banner(s)
    character(*), intent(in) :: s
    print '(a)', repeat('-', 8)//' '//trim(s)//' '//repeat('-', 8)
  end subroutine banner

  subroutine assert_true(cond, msg, ok)
    logical, intent(in)  :: cond
    character(*), intent(in) :: msg
    logical, intent(inout) :: ok
    if (.not. cond) then
      ok = .false.
      write(stderr,'(a)') trim(msg)
    end if
  end subroutine assert_true

  subroutine set_input3d(lay, x)
    type(layer), intent(inout) :: lay
    real, intent(in) :: x(:,:,:)
    select type(p => lay%p)
    type is (input3d_layer)
      call p%set(x)
    class default
      call bail("expected input3d_layer in set_input3d")
    end select
  end subroutine set_input3d

  subroutine set_input2d(lay, x)
    type(layer), intent(inout) :: lay
    real, intent(in) :: x(:,:)
    select type(p => lay%p)
    type is (input2d_layer)
      call p%set(x)
    class default
      call bail("expected input2d_layer in set_input2d")
    end select
  end subroutine set_input2d

  subroutine grab_flatten_gradients3d(lay, g)
    type(layer), intent(in) :: lay
    real, allocatable, intent(out) :: g(:,:,:)
    select type(p => lay%p)
    type is (flatten_layer)
      if (allocated(p%gradient_3d)) then
        g = p%gradient_3d
      else
        call bail("flatten_layer%gradient_3d is not allocated")
      end if
    class default
      call bail("expected flatten_layer in grab_flatten_gradients3d")
    end select
  end subroutine grab_flatten_gradients3d

  subroutine grab_flatten_gradients2d(lay, g)
    type(layer), intent(in) :: lay
    real, allocatable, intent(out) :: g(:,:)
    select type(p => lay%p)
    type is (flatten_layer)
      if (allocated(p%gradient_2d)) then
        g = p%gradient_2d
      else
        call bail("flatten_layer%gradient_2d is not allocated")
      end if
    class default
      call bail("expected flatten_layer in grab_flatten_gradients2d")
    end select
  end subroutine grab_flatten_gradients2d

  subroutine bail(msg)
    character(*), intent(in) :: msg
    write(stderr,'(a)') trim(msg)
    error stop 2
  end subroutine bail

end program test_flatten_layer
