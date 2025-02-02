program test_linear2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_linear2d_layer, only: linear2d_layer
  implicit none

  logical :: ok = .true.
  real :: sample_input(2, 3, 4) = reshape(&
      [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2,&
       0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],&
      [2, 3, 4]) ! first batch are 0.1, second 0.2
  real :: sample_gradient(2, 3, 1) = reshape([2., 2., 2., 2., 2., 2.], [2, 3, 1])
  type(linear2d_layer) :: linear

  linear = linear2d_layer(batch_size=2, sequence_length=3, in_features=4, out_features=1)

  call test_linear2d_layer_forward(linear, ok, sample_input)
  call test_linear2d_layer_backward(linear, ok, sample_input, sample_gradient)

contains
  subroutine test_linear2d_layer_forward(linear, ok, input)
    type(linear2d_layer), intent(in out) :: linear
    logical, intent(in out) :: ok
    real, intent(in) :: input(2, 3, 4)
    real :: output_shape(3)
    real :: output_flat(6)
    real :: expected_shape(3) = [2, 3, 1]
    real :: expected_output_flat(6) = [0.15, 0.19, 0.15, 0.19, 0.15, 0.19]

    call linear % forward(input)

    output_shape = shape(linear % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
    output_flat = reshape(linear % output, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if
  end subroutine test_linear2d_layer_forward

  subroutine test_linear2d_layer_backward(linear, ok, input, gradient)
    type(linear2d_layer), intent(in out) :: linear
    logical, intent(in out) :: ok
    real, intent(in) :: input(2, 3, 4)
    real, intent(in) :: gradient(2, 3, 1)
    real :: gradient_shape(3)
    real :: dw_shape(2)
    real :: db_shape(1)
    real :: gradient_flat(24)
    real :: dw_flat(4)
    real :: expected_gradient_shape(3) = [2, 3, 4]
    real :: expected_dw_shape(2) = [4, 1]
    real :: expected_db_shape(1) = [1]
    real :: expected_gradient_flat(24)
    real :: expected_dw_flat(4)
    real :: expected_db(1) = [12.0]

    expected_gradient_flat = 0.200000003
    expected_dw_flat = 1.80000007

    call linear % backward(input, gradient)

    gradient_shape = shape(linear % gradient)
    if (.not. all(gradient_shape.eq.expected_gradient_shape)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect gradient shape.. failed'
    end if
    dw_shape = shape(linear % dw)
    if (.not. all(dw_shape.eq.expected_dw_shape)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect dw shape.. failed'
    end if
    db_shape = shape(linear % db)
    if (.not. all(db_shape.eq.expected_db_shape)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect db shape.. failed'
    end if

    gradient_flat = reshape(linear % gradient, shape(gradient_flat))
    if (.not. all(gradient_flat.eq.expected_gradient_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect gradient values.. failed'
    end if
    dw_flat = reshape(linear % dw, shape(dw_flat))
    if (.not. all(dw_flat.eq.expected_dw_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect dw values.. failed'
    end if
    if (.not. all(linear % db.eq.expected_db)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect db values.. failed'
    end if
  end subroutine test_linear2d_layer_backward
end program test_linear2d_layer