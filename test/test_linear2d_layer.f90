program test_linear2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_linear2d_layer, only: linear2d_layer
  implicit none

  logical :: ok = .true.
  real :: sample_input(3, 4) = reshape(&
      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],&
      [3, 4]) ! first batch are 0.1, second 0.2
  real :: sample_gradient(3, 1) = reshape([2., 2., 3.], [3, 1])
  type(linear2d_layer) :: linear

  linear = linear2d_layer(sequence_length=3, in_features=4, out_features=1)
  call linear % init([4])

  call test_linear2d_layer_forward(linear, ok, sample_input)
  call test_linear2d_layer_backward(linear, ok, sample_input, sample_gradient)
  call test_linear2d_layer_gradient_updates(ok)

contains
  subroutine test_linear2d_layer_forward(linear, ok, input)
    type(linear2d_layer), intent(in out) :: linear
    logical, intent(in out) :: ok
    real, intent(in) :: input(3, 4)
    real :: output_shape(2)
    real :: output_flat(3)
    real :: expected_shape(2) = [3, 1]
    real :: expected_output_flat(3) = [0.17, 0.17, 0.17]

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
    real, intent(in) :: input(3, 4)
    real, intent(in) :: gradient(3, 1)
    real :: gradient_shape(2)
    real :: dw_shape(2)
    real :: db_shape(1)
    real :: gradient_flat(12)
    real :: dw_flat(4)
    real :: expected_gradient_shape(2) = [3, 4]
    real :: expected_dw_shape(2) = [4, 1]
    real :: expected_db_shape(1) = [1]
    real :: expected_gradient_flat(12) = [&
        0.2, 0.2, 0.3, 0.2,&
        0.2, 0.3, 0.2, 0.2,&
        0.3, 0.2, 0.2, 0.3&
    ]
    real :: expected_dw_flat(4) = [0.7, 0.7, 1.4, 1.4]
    real :: expected_db(1) = [7]

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

  subroutine test_linear2d_layer_gradient_updates(ok)
    logical, intent(in out) :: ok
    real :: input(3, 4) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [3, 4])
    real :: gradient(3, 2) = reshape([0.0, 10., 0.2, 3., 0.4, 1.], [3, 2])
    type(linear2d_layer) :: linear

    integer :: num_parameters
    real :: parameters(10)
    real :: expected_parameters(10) = [&
        0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001,&
        0.109999999, 0.109999999&
    ]
    real :: gradients(10)
    real :: expected_gradients(10) = [&
        1.03999996, 4.09999990, 7.15999985, 1.12400007, 0.240000010, 1.56000006, 2.88000011, 2.86399961,&
        10.1999998, 4.40000010&
    ]
    real :: updated_parameters(10)
    real :: updated_weights(8)
    real :: updated_biases(2)
    real :: expected_weights(8) = [&
        0.203999996, 0.509999990, 0.816000044, 0.212400019, 0.124000005, 0.256000012, 0.388000011, 0.386399955&
    ]
    real :: expected_biases(2) = [1.13000000, 0.550000012]

    integer :: i

    linear = linear2d_layer(sequence_length=3, in_features=4, out_features=2, batch_size=1)
    call linear % init([4])
    call linear % forward(input)
    call linear % backward(input, gradient)

    num_parameters = linear % get_num_params()
    if (num_parameters /= 10) then
      ok = .false.
      write(stderr, '(a)') 'incorrect number of parameters.. failed'
    end if

    parameters = linear % get_params()
    if (.not. all(parameters.eq.expected_parameters)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect parameters.. failed'
    end if

    gradients = linear % get_gradients()
    if (.not. all(gradients.eq.expected_gradients)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect gradients.. failed'
    end if

    do i = 1, num_parameters
      updated_parameters(i) = parameters(i) + 0.1 * gradients(i)
    end do
    call linear % set_params(updated_parameters)
    updated_weights = reshape(linear % weights, shape(expected_weights))
    if (.not. all(updated_weights.eq.expected_weights)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect updated weights.. failed'
    end if
    updated_biases = linear % biases
    if (.not. all(updated_biases.eq.expected_biases)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect updated biases.. failed'
    end if
  end subroutine test_linear2d_layer_gradient_updates
end program test_linear2d_layer