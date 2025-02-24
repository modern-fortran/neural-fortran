program test_fc2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_fc2d_layer, only: fc2d_layer
  use nf, only: activation_function, relu, tanhf, sigmoid, softplus, sgd
  implicit none

  logical :: ok = .true.
  real :: sample_input(3, 4) = reshape(&
      [0.0, -10.1, 0.2, 10.3, 0.4, 10.5, -0.6, 10.7, 10.8, 0.9, 0.11, 0.12],&
      [3, 4])
  real :: sample_gradient(3, 4) = reshape([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.], [3, 4])
  type(fc2d_layer) :: fc

  call test_fc2d_layer_forward(ok, sample_input)
  call test_fc2d_layer_backward(&
    ok, sample_input, sample_gradient,&
    activation=relu(),&
    expected_gradient_flat=[&
      0.198, 0.486, 0.486,&
      0.396, 0.972, 0.972,&
      0.594, 1.458, 1.458,&
      0.792, 1.944, 1.944&
    ]&
  )
  call test_fc2d_layer_backward(&
    ok, sample_input, sample_gradient,&
    activation=sigmoid(),&
    expected_gradient_flat=[&
      0.01068044, 0.02734236, 0.00086295,&
      0.02136087, 0.05140798, 0.00172666,&
      0.03357822, 0.07555774, 0.00266102,&
      0.04567052, 0.10338347, 0.0038053&
    ]&
  )
  call test_fc2d_layer_backward(&
    ok, sample_input, sample_gradient,&
    activation=tanhf(),&
    expected_gradient_flat=[&
      3.7096841e-03, 9.3461145e-03, 1.1113838e-05,&
      7.4193683e-03, 1.6985621e-02, 2.2227676e-05,&
      1.2096796e-02, 2.4647098e-02, 3.3862932e-05,&
      1.6650427e-02, 3.4423053e-02, 5.0007438e-05&
    ]&
  )
  call test_fc2d_layer_backward(&
    ok, sample_input, sample_gradient,&
    activation=softplus(),&
    expected_gradient_flat=[&
      0.18651924, 0.45662752, 0.48513436,&
      0.37303847, 0.9168981, 0.9702679,&
      0.5578177, 1.3770795, 1.4553307,&
      0.7427467, 1.8331366, 1.9401824&
    ]&
  )
  call test_fc2d_layer_update_gradients(ok, sample_input, sample_gradient)

  if (ok) then
    print '(a)', 'test_fc2d_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_fc2d_layer: One or more tests failed.'
    error stop 1
  end if

contains
  function allclose(x, y) result(res)
    real, intent(in) :: x(:)
    real, intent(in) :: y(:)
    logical :: res

    res = all(abs(x - y) <= (1e-06 + 1e-05 * abs(y)))
  end function allclose

  subroutine init_weigths(fc)
    type(fc2d_layer) :: fc
    fc % in_proj % weights = reshape(&
        [&
            0.1, 0.2, 0.3, 0.4, 0.1,&
            0.2, 0.3, 0.5, 0.1, 0.2,&
            0.4, 0.5, 0.1, 0.3, 0.4,&
            0.5, 0.2, 0.3, 0.4, 0.5&
        ],&
        [4, 5]&
    )
    fc % in_proj % biases = 0.11
    fc % out_proj % weights = 0.1
    fc % out_proj % biases = 0.11
  end subroutine init_weigths

  subroutine test_fc2d_layer_forward(ok, input)
    logical, intent(in out) :: ok
    real, intent(in) :: input(3, 4)
    type(fc2d_layer) :: fc
    real :: output_shape(2)
    real :: output_flat(12)
    real :: expected_shape(2) = [3, 4]
    real :: expected_output_flat(12) = [&
        1.509, 1.5594, 3.4098,&
        1.509, 1.5594, 3.4098,&
        1.509, 1.5594, 3.4098,&
        1.509, 1.5594, 3.4098&
    ]

    fc = fc2d_layer(hidden_size=5, activation=relu())
    call fc % init([3, 4])
    call init_weigths(fc)

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

  subroutine test_fc2d_layer_backward(ok, input, gradient, activation, expected_gradient_flat)
    logical, intent(in out) :: ok
    real, intent(in) :: input(3, 4)
    real, intent(in) :: gradient(3, 4)
    class(activation_function), intent(in) :: activation
    real, intent(in) :: expected_gradient_flat(12)

    type(fc2d_layer) :: fc

    integer :: gradient_shape(2)
    integer :: expected_gradient_shape(2) = [3, 4]
    real :: gradient_flat(12)

    fc = fc2d_layer(hidden_size=5, activation=activation)
    call fc % init([3, 4])
    call init_weigths(fc)

    call fc % forward(input)
    call fc % backward(input, gradient)

    gradient_shape = shape(fc % gradient)
    if (.not. all(gradient_shape.eq.expected_gradient_shape)) then
      ok = .false.
      write(stderr, '(aa)') 'backward returned incorrect gradient shape.. failed for', fc % activation % get_name()
    end if
    gradient_flat = reshape(fc % gradient, shape(gradient_flat))
    if (.not. allclose(gradient_flat, expected_gradient_flat)) then
      ok = .false.
      write(stderr, '(aa)') 'backward returned incorrect gradient values.. failed for ', fc % activation % get_name()
    end if
  end subroutine test_fc2d_layer_backward

  subroutine test_fc2d_layer_update_gradients(ok, input, gradient)
    logical, intent(in out) :: ok
    real, intent(in) :: input(3, 4)
    real, intent(in) :: gradient(3, 4)

    type(fc2d_layer) :: fc
    type(sgd) :: optim

    real :: parameters(49)
    real :: updated_output(12)
    real :: expected_updated_output(12) = [&
        -1.1192487, -0.51458186, -2.2737966,&
        -1.7527609, -0.8190526, -3.5071785,&
        0.36815026, 0.2097921, 0.6197472,&
        -1.7491575, -0.79099315, -3.4819508&
    ]

    fc = fc2d_layer(hidden_size=5, activation=softplus())
    call fc % init([3, 4])
    call init_weigths(fc)

    call fc % forward(input)
    call fc % backward(input, gradient)

    if (fc % get_num_params() /= 49) then
      ok = .false.
      write(stderr, '(a)') 'incorrect number of parameters.. failed'
    end if

    optim = SGD(learning_rate=0.01)
    parameters = fc % get_params()
    call optim % minimize(parameters, fc % get_gradients())
    call fc % set_params(parameters)

    call fc % forward(input)

    updated_output = reshape(fc % output, [12])
    if (.not. allclose(updated_output, expected_updated_output)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect output after parameters update.. failed'
    end if
  end subroutine test_fc2d_layer_update_gradients
end program test_fc2d_layer
