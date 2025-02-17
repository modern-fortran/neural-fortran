program test_layernorm
  use iso_fortran_env, only: stderr => error_unit
  use nf_layernorm_layer, only: layernorm_layer
  implicit none

  logical :: ok = .true.
  type(layernorm_layer) :: layernorm
  real :: sample_input(3, 4) = reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4])
  real :: sample_gradient(3, 4) = reshape([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.], [3, 4])

  layernorm = layernorm_layer()
  call layernorm % init([3, 4])

  call test_layernorm_forward(layernorm, sample_input, ok)
  call test_layernorm_backward(layernorm, sample_input, sample_gradient, ok)

  if (ok) then
    print '(a)', 'test_layernorm_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_layernorm_layer: One or more tests failed.'
    stop 1
  end if

contains
  subroutine test_layernorm_forward(layernorm, input, ok)
    type(layernorm_layer), intent(in out) :: layernorm
    real, intent(in out) :: input(:, :)
    logical, intent(in out) :: ok
    real :: output_shape(2)
    real :: output_flat(12)
    real :: expected_shape(2) = [3, 4]
    real :: expected_output_flat(12) = [&
        -0.693158746, 0.939844191, -0.992156327, 1.72702277, -0.970368207, 0.971188426,&
        -0.552177250, 1.05800152, 1.02837324, -0.481686622, -1.02747762, -1.00740564&
    ]

    call layernorm % forward(input)

    output_shape = shape(layernorm % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
    output_flat = reshape(layernorm % output, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if
  end subroutine test_layernorm_forward

  subroutine test_layernorm_backward(layernorm, input, gradient, ok)
    type(layernorm_layer), intent(in out) :: layernorm
    real, intent(in out) :: input(:, :)
    real, intent(in out) :: gradient(:, :)
    logical, intent(in out) :: ok

    real :: gradient_shape(2)
    real :: gradient_flat(12)
    real :: expected_gradient_shape(2) = [3, 4]
    real :: expected_gradient_flat(12) = [&
        -0.227230772, 0.103088334, -9.88590196E-02, -2.86390483E-02, 0.283811331, 0.277955681,&
        -0.215662330, -0.105019525, -0.269407451, 0.471532196, -0.281880081, 9.03107598E-02&
    ]

    real :: d_gamma(4)
    real :: expected_d_gamma(4) = [0.765904069, 0.175162792,  2.16362262, -4.57002449]
    real :: d_beta(4)
    real :: expected_d_beta(4) = [5.09999990, 6.09999990, 2.19999981, 6.09999990]

    call layernorm % backward(input, gradient)

    gradient_shape = shape(layernorm % gradient)
    if (.not. all(gradient_shape.eq.expected_gradient_shape)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect gradient shape.. failed'
    end if
    gradient_flat = reshape(layernorm % gradient, shape(gradient_flat))
    if (.not. all(gradient_flat.eq.expected_gradient_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect gradient values.. failed'
    end if

    if (.not. all(layernorm % d_gamma.eq.expected_d_gamma)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect d_gamma values.. failed'
    end if
    if (.not. all(layernorm % d_beta.eq.expected_d_beta)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect d_beta values.. failed'
    end if
  end subroutine test_layernorm_backward

end program test_layernorm
