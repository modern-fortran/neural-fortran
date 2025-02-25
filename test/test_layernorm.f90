program test_layernorm_instance
  use iso_fortran_env, only: stderr => error_unit
  use nf_layernorm_layer, only: layernorm_layer
  use nf_linear2d_layer, only: linear2d_layer
  use nf_layer, only: layer
  use nf, only: sgd, layernorm, network, input, flatten, linear2d
  implicit none

  logical :: ok = .true.
  type(layernorm_layer) :: layernorm_instance
  real :: sample_input(3, 4) = reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4])
  real :: sample_gradient(3, 4) = reshape([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.], [3, 4])

  layernorm_instance = layernorm_layer()
  call layernorm_instance % init([3, 4])

  call test_layernorm_forward(layernorm_instance, sample_input, ok)
  call test_layernorm_backward(layernorm_instance, sample_input, sample_gradient, ok)
  call test_layernorm_gradients(sample_input, sample_gradient, ok)
  call test_layernorm_integration(ok)

  if (ok) then
    print '(a)', 'test_layernorm_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_layernorm_layer: One or more tests failed.'
    error stop 1
  end if

contains
  function allclose(x, y) result(res)
    real, intent(in) :: x(:)
    real, intent(in) :: y(:)
    logical :: res

    res = all(abs(x - y) <= (1e-06 + 1e-05 * abs(y)))
  end function allclose

  subroutine test_layernorm_forward(layernorm_instance, input, ok)
    type(layernorm_layer), intent(in out) :: layernorm_instance
    real, intent(in out) :: input(:, :)
    logical, intent(in out) :: ok
    real :: output_shape(2)
    real :: output_flat(12)
    real :: expected_shape(2) = [3, 4]
    real :: expected_output_flat(12) = [&
        -0.693158746, 0.939844191, -0.992156327, 1.72702277, -0.970368207, 0.971188426,&
        -0.552177250, 1.05800152, 1.02837324, -0.481686622, -1.02747762, -1.00740564&
    ]

    call layernorm_instance % forward(input)

    output_shape = shape(layernorm_instance % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
    output_flat = reshape(layernorm_instance % output, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if
  end subroutine test_layernorm_forward

  subroutine test_layernorm_backward(layernorm_instance, input, gradient, ok)
    type(layernorm_layer), intent(in out) :: layernorm_instance
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
    real :: expected_d_beta(4) = [5.1, 6.1, 2.2, 6.1]

    call layernorm_instance % backward(input, gradient)

    gradient_shape = shape(layernorm_instance % gradient)
    if (.not. all(gradient_shape.eq.expected_gradient_shape)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect gradient shape.. failed'
    end if
    gradient_flat = reshape(layernorm_instance % gradient, shape(gradient_flat))
    if (.not. allclose(gradient_flat, expected_gradient_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect gradient values.. failed'
    end if

    if (.not. allclose(layernorm_instance % d_gamma, expected_d_gamma)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect d_gamma values.. failed'
    end if
    if (.not. allclose(layernorm_instance % d_beta, expected_d_beta)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect d_beta values.. failed'
    end if
  end subroutine test_layernorm_backward

  subroutine test_layernorm_gradients(input, gradient, ok)
    real, intent(in out) :: input(:, :)
    real, intent(in out) :: gradient(:, :)
    logical, intent(in out) :: ok
    type(layernorm_layer) :: layernorm_instance
    type(sgd) :: optim

    real :: parameters(8)
    real :: expected_parameters(8)
    real :: updated_output(12)
    real :: expected_updated_output(12) = [&
        -0.738849819, 0.881645918, -1.03555739,&
        1.66299772, -1.02966857, 0.908487320,&
        -0.562230229, 1.01311040, 0.984123051,&
        -0.564699769, -1.13543355, -1.11444426&
    ]

    layernorm_instance = layernorm_layer()
    call layernorm_instance % init([3, 4])

    call layernorm_instance % forward(input)
    call layernorm_instance % backward(input, gradient)

    if (layernorm_instance % get_num_params() /= 8) then
      ok = .false.
      write(stderr, '(a)') 'incorrect number of parameters.. failed'
    end if

    expected_parameters(1: 4) = 1.
    expected_parameters(5: 8) = 0.
    parameters = layernorm_instance % get_params()
    if (.not. all(parameters.eq.expected_parameters)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect parameters.. failed'
    end if

    optim = SGD(learning_rate=0.01)
    call optim % minimize(parameters, layernorm_instance % get_gradients())
    call layernorm_instance % set_params(parameters)

    call layernorm_instance % forward(input)

    updated_output = reshape(layernorm_instance % output, [12])
    if (.not. allclose(updated_output, expected_updated_output)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect output after parameters update.. failed'
    end if
  end subroutine test_layernorm_gradients

  subroutine test_layernorm_integration(ok)
    logical, intent(in out) :: ok

    type(network) :: net
    real :: x(2, 3) = reshape([0.1, 2., 0.3, 4., 0.5, 6.], [2, 3])
    real :: y(6) = [0.7, 0.2, 0.1, 0.1, 0.01, 0.9]
    real :: tolerance = 0.1
    integer :: epoch
    integer :: epochs = 10000

    net = network([&
        input(2, 3),&
        linear2d(3),&
        layernorm(),&
        flatten()&
    ])

    ! Kaiming weights to achieve semblance of convergance
    select type(l => net % layers(2) % p)
      type is(linear2d_layer)
      call random_number(l % weights)
      l % weights = l % weights * sqrt(2. / 6.)
      l % biases = 0.2
    end select

    do epoch = 1, epochs
      call net % forward(x)
      call net % backward(y)
      call net % update(optimizer=sgd(learning_rate=0.001))
      if (all(abs(net % predict(x) - y) < tolerance)) exit
    end do
    print *, abs(net % predict(x) - y)

    print *, epoch
    if (.not. epoch <= epochs) then
      write(stderr, '(a)') &
        'linear2d + layernorm should converge in simple training.. failed'
      ok = .false.
    end if
  end subroutine test_layernorm_integration
end program test_layernorm_instance
