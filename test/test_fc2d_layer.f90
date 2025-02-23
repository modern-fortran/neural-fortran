program test_fc2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_fc2d_layer, only: fc2d_layer
  use nf, only: relu
  implicit none

  logical :: ok = .true.
  real :: sample_input(3, 4) = reshape(&
      [0.0, -10.1, 0.2, 10.3, 0.4, 10.5, -0.6, 10.7, 10.8, 0.9, 0.11, 0.12],&
      [3, 4])
  real :: sample_gradient(3, 4) = reshape([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.], [3, 4])
  type(fc2d_layer) :: fc

  fc = fc2d_layer(hidden_size=5, activation=relu())
  call fc % init([3, 4])
  fc % in_proj % weights = 0.1
  fc % in_proj % biases = 0.11
  fc % out_proj % weights = 0.1
  fc % out_proj % biases = 0.11

  call test_fc2d_layer_forward(fc, ok, sample_input)

  if (ok) then
    print '(a)', 'test_fc2d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_fc2d_layer: One or more tests failed.'
    stop 1
  end if

contains
  function allclose(x, y) result(res)
    real, intent(in) :: x(:)
    real, intent(in) :: y(:)
    logical :: res

    res = all(abs(x - y) <= (1e-06 + 1e-05 * abs(y)))
  end function allclose

  subroutine test_fc2d_layer_forward(fc, ok, input)
    type(fc2d_layer), intent(in out) :: fc
    logical, intent(in out) :: ok
    real, intent(in) :: input(3, 4)
    real :: output_shape(2)
    real :: output_flat(12)
    real :: expected_shape(2) = [3, 4]
    real :: expected_output_flat(12) = [&
        0.695, 0.2205, 1.246,&
        0.695, 0.2205, 1.246,&
        0.695, 0.2205, 1.246,&
        0.695, 0.2205, 1.246&
    ]

    call fc % forward(input)

    output_shape = shape(fc % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
    output_flat = reshape(fc % output, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if
  end subroutine test_fc2d_layer_forward

end program test_fc2d_layer