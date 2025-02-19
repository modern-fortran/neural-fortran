program test_multihead_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_multihead_attention_layer, only: multihead_attention_layer
  use nf_self_attention_layer, only: self_attention_layer
  use nf_cross_attention_layer, only: cross_attention_layer
  use nf_linear2d_layer, only: linear2d_layer
  use nf_optimizers, only: sgd
  implicit none

  logical :: ok = .true.
  type(multihead_attention_layer) :: attention
  real :: sample_input(3, 4) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [3, 4])
  real :: split_heads_output(3, 2, 2)
  real :: minput(3, 4) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [3, 4])
  real :: output(3, 2, 2)

  attention = multihead_attention_layer(n_heads=2)
  call attention % init_base([3, 4])
  call set_weights(attention)

  call test_multihead_attention_split_heads(attention, sample_input, ok, split_heads_output)
  call test_multihead_attention_create_attention_matrix(attention, split_heads_output, ok)
  call test_multihead_attention_normalization(attention, ok)
  call test_multihead_attention_scaled_dot_product_attention(attention, split_heads_output, ok)
  call test_multihead_attention_combine_heads(attention, attention % sdpa, ok)
  call test_multihead_attention_forward(attention, ok)
  call test_multihead_attention_backward(attention, ok)
  call test_multihead_attention_update_gradients(attention, ok)
  call test_multihead_attention_forward_reallife_shape(ok)
  call test_self_attention(ok)
  call test_cross_attention(ok)

contains
  function allclose(x, y) result(res)
    real, intent(in) :: x(:)
    real, intent(in) :: y(:)
    logical :: res

    res = all(abs(x - y) <= (1e-08 + 1e-05 * abs(y)))
  end function allclose

  subroutine set_weights(attention)
    type(multihead_attention_layer), intent(in out) :: attention
    attention % query_layer % weights = 0.1
    attention % key_layer % weights = 0.1
    attention % value_layer % weights = 0.1
    attention % output_layer % weights = 0.1
    attention % query_layer % biases = 0.11
    attention % key_layer % biases = 0.11
    attention % value_layer % biases = 0.11
    attention % output_layer % biases = 0.11
  end subroutine set_weights

  subroutine test_multihead_attention_split_heads(attention, input, ok, output)
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: input(:, :)
    logical, intent(in out) :: ok
    real, intent(in out) :: output(3, 2, 2)
    real :: output_shape(3)
    real :: expected_shape(3) = [3, 2, 2]
    real :: output_flat(12)
    real :: expected_output_flat(12) = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12]

    output = attention % split_heads(input)

    output_shape = shape(output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'split_heads returned incorrect shape.. failed'
    end if
    output_flat = reshape(output, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'split_heads returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_split_heads

  subroutine test_multihead_attention_create_attention_matrix(attention, input, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    real, intent(in) :: input(:, :, :)
    logical, intent(in out) :: ok
    real :: attention_matrix_shape(3)
    real, volatile :: attention_matrix_flat(18)
    real :: expected_shape(3) = [3, 3, 2]
    real :: expected_attention_matrix_flat(18) = [&
        0.09, 0.12, 0.15, 0.12, 0.17, 0.22,&
        0.15, 0.22, 0.29, 1.17, 0.519, 0.588,&
        0.519, 0.5021, 0.5732, 0.588, 0.5732, 0.6544&
    ]

    call attention % create_attention_matrix(input, input)
    print *, attention % attention_matrix

    attention_matrix_shape = shape(attention % attention_matrix)
    if (.not. all(attention_matrix_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'create_attention_matrix returned incorrect shape.. failed'
    end if
    attention_matrix_flat = reshape(attention % attention_matrix, shape(expected_attention_matrix_flat))
    if (.not. allclose(attention_matrix_flat, expected_attention_matrix_flat)) then
      ok = .false.
      write(stderr, '(a)') 'create_attention_matrix returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_create_attention_matrix

  subroutine test_multihead_attention_normalization(attention, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    logical, intent(in out) :: ok
    real, volatile :: output_flat(18)
    real :: expected_output_flat(18) = [&
        0.326287806, 0.321620107, 0.316976935, 0.333283335, 0.333194494, 0.333061278,&
        0.340428889, 0.345185429, 0.349961787, 0.435975075, 0.330339372, 0.329200655,&
        0.275134116, 0.326415271, 0.325773478, 0.288890868, 0.343245387, 0.345025837&
    ]

    call attention % normalize_attention_matrix()

    output_flat = reshape(attention % attention_matrix, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'normalize_attention_matrix returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_normalization

  subroutine test_multihead_attention_scaled_dot_product_attention(attention, value, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    real, intent(in) :: value(:, :, :)
    logical, intent(in out) :: ok
    real, volatile :: output_flat(12)
    real :: expected_output_flat(12) = [&
        0.101414114, 0.102356538, 0.103298485, 0.401414126, 0.402356565, 0.403298497,&
        0.685291648, 0.701290667, 0.701582491, 0.457309216, 0.374400556, 0.373518765&
    ]

    call attention % scaled_dot_product_attention(value)

    output_flat = reshape(attention % sdpa, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'scaled_dot_product_attention returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_scaled_dot_product_attention

  subroutine test_multihead_attention_combine_heads(attention, scaled_dp_att, ok)
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: scaled_dp_att(:, :, :)
    logical, intent(in out) :: ok
    real :: output(attention % sequence_length, attention % model_dimension)
    real :: output_flat(12)
    real :: expected_output_flat(12) = [&
        0.101414114, 0.102356538, 0.103298485, 0.685291648, 0.701290667, 0.701582491,&
        0.401414126, 0.402356565, 0.403298497, 0.457309216, 0.374400556, 0.373518765&
    ]

    output = attention % combine_heads(scaled_dp_att)

    output_flat = reshape(output, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'combine_heads returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_combine_heads

  subroutine test_multihead_attention_forward(attention, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    logical, intent(in out) :: ok
    real :: input(3, 4) = reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4])
    real :: output(attention % sequence_length, attention % model_dimension)
    real, volatile :: output_flat(12)
    integer :: output_shape(2)
    integer :: attn_weights_shape(3)
    real, volatile :: attn_weights_flat(18)
    integer :: expected_shape(2) = [3, 4]
    real :: expected_output_flat(12) = [&
        0.982241452, 1.00407875, 1.00444126, 0.982241452, 1.00407875, 1.00444126,&
        0.982241452, 1.00407875, 1.00444126, 0.982241452, 1.00407875, 1.00444126&
    ]
    integer :: expected_attn_weights_shape(3) = [3, 3, 2]
    real :: expected_attn_weights_flat(18) = [&
        7.89450705E-02, 2.28110179E-02, 2.18846574E-02, 0.447508544, 0.464612424, 0.464721352,&
        0.473546445, 0.512576580, 0.513393998, 7.89450705E-02, 2.28110179E-02, 2.18846574E-02,&
        0.447508544, 0.464612424, 0.464721352, 0.473546445, 0.512576580, 0.513393998&
    ]

    call attention % common_forward(input, input, input)

    output_shape = shape(attention % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
    output_flat = reshape(attention % output, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if

    attn_weights_shape = shape(attention % attention_matrix)
    if (.not. all(attn_weights_shape.eq.expected_attn_weights_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect attention weights shape.. failed'
    end if
    attn_weights_flat = reshape(attention % attention_matrix, shape(attn_weights_flat))
    if (.not. allclose(attn_weights_flat, expected_attn_weights_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect attention weights values.. failed'
    end if
  end subroutine test_multihead_attention_forward

  subroutine test_multihead_attention_forward_reallife_shape(ok)
    logical, intent(in out) :: ok
    real :: input(148, 512)
    real :: output(148, 512)
    integer :: output_shape(2)
    integer :: expected_shape(2) = [148, 512]
    type(multihead_attention_layer) :: attention

    call random_number(input)

    attention = multihead_attention_layer(n_heads=8)
    call attention % init_base([148, 512])
    call set_weights(attention)

    call attention % common_forward(input, input, input)

    output_shape = shape(attention % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
  end subroutine test_multihead_attention_forward_reallife_shape

  subroutine test_multihead_attention_backward(attention, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    logical, intent(in out) :: ok
    real :: input(3, 4) = reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4])
    real :: gradient(3, 4) = reshape([0.1, 3., 2., 0.1, 3., 3., 0.1, 2., 0.1, 3., 0.1, 3.], [3, 4])
    real :: expected_output_flat(12) = [&
        -2.29912549E-02, 0.381484956, 0.453185737,&
        -2.29912549E-02, 0.381484956, 0.453185737,&
        -2.29912549E-02, 0.381484956, 0.453185737,&
        -2.29912549E-02, 0.381484956, 0.453185737&
    ]
    real :: expected_shape(2) = [3, 4]
    real :: output(3, 4)
    real, volatile :: output_flat(12)
    real :: output_shape(2)

    call attention % common_backward(input, gradient)

    ! sample for Self Attention: sum of output gradients
    ! FIXME: remove reshapes when linear2d situation is resolved
    output = &
        reshape(attention % query_layer % gradient, [attention % sequence_length, attention % model_dimension]) &
        + reshape(attention % key_layer % gradient, [attention % sequence_length, attention % model_dimension]) &
        + reshape(attention % value_layer % gradient, [attention % sequence_length, attention % model_dimension])

    output_shape = shape(output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect shape.. failed'
    end if
    output_flat = reshape(output, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_backward

  subroutine test_multihead_attention_update_gradients(attention, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    logical, intent(in out) :: ok
    real :: parameters(80)
    real :: expected_parameters(80)
    real, volatile :: updated_output(12)
    real :: expected_updated_output(12) = [&
        0.111365855, 0.115744293, 0.115733206, 0.185253710, 0.196646214, 0.196617395,&
        -0.102874994, -0.118834510, -0.118794113, 0.179314315, 0.190210193, 0.190182626&
    ]
    type(sgd) :: optim

    if (attention % get_num_params() /= 80) then
      ok = .false.
      write(stderr, '(a)') 'incorrect number of parameters.. failed'
    end if

    expected_parameters(1: 64) = 0.100000001
    expected_parameters(65: 80) = 0.109999999
    parameters = attention % get_params()
    if (.not. all(parameters.eq.expected_parameters)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect parameters.. failed'
    end if

    optim = SGD(learning_rate=0.01)
    call optim % minimize(parameters, attention % get_gradients())
    call attention % set_params(parameters)

    call attention % common_forward(&
        reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4]),&
        reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4]),&
        reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4])&
    )

    updated_output = reshape(attention % output, [12])
    if (.not. allclose(updated_output, expected_updated_output)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect output after parameters update.. failed'
    end if
  end subroutine test_multihead_attention_update_gradients

  subroutine test_self_attention(ok)
    logical, intent(in out) :: ok
    type(self_attention_layer) :: attention
    real :: input(2, 3) = reshape([-1., 0., 17., .4, 5., .6], [2, 3])
    real :: output(2, 3)
    real, volatile :: output_flat(6)
    real :: expected_output_flat(6) = [&
        0.772716165, 0.577548742, 0.772716165, 0.577548742, 0.772716165, 0.577548742&
    ]
    real :: gradient(2, 3) = reshape([1., 2., .17, 4., .5, 6.], [2, 3])
    real, volatile :: gradient_flat(6)
    real :: expected_gradient_flat(6) = [&
        0.350671142, 0.607403040, 0.350671142, 0.607403040, 0.350671142, 0.607403040&
    ]

    attention = self_attention_layer(n_heads=1)
    call attention % init([2, 3])
    attention % query_layer % weights = 0.1
    attention % key_layer % weights = 0.1
    attention % value_layer % weights = 0.1
    attention % output_layer % weights = 0.1
    attention % query_layer % biases = 0.11
    attention % key_layer % biases = 0.11
    attention % value_layer % biases = 0.11
    attention % output_layer % biases = 0.11

    call attention % forward(input)
    output_flat = reshape(attention % output, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if

    call attention % backward(input, gradient)
    gradient_flat = reshape(attention % gradient, shape(gradient_flat))
    if (.not. allclose(gradient_flat, expected_gradient_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect values.. failed'
    end if
  end subroutine test_self_attention

  subroutine test_cross_attention(ok)
    logical, intent(in out) :: ok
    type(cross_attention_layer) :: attention
    real :: query(2, 3) = reshape([-1., 0., 17., .4, 5., .6], [2, 3])
    real :: key_value(2, 3) = reshape([0.1, -.2, 0.3, 4., 15., 0.5], [2, 3])
    real :: input(2, 2, 3)
    real :: output(2, 2, 3)
    real, volatile :: output_flat(6)
    real :: expected_output_flat(6) = [&
        0.600311756, 0.471662223, 0.600311756, 0.471662223, 0.600311756, 0.471662223&
    ]
    real :: gradient(2, 3) = reshape([1., 2., .17, 4., .5, 6.], [2, 3])
    real, volatile :: query_gradient_flat(6)
    real, volatile :: key_value_gradient_flat(6)
    real :: expected_query_gradient_flat(6) = [&
        1.48406753E-03, 0.184446245, 1.48406753E-03, 0.184446245, 1.48406753E-03, 0.184446245&
    ]
    real :: expected_key_value_gradient_flat(6) = [&
        0.303095698, 0.107004307, 0.303095698, 0.107004307, 0.303095698, 0.107004307&
    ]
    input(1, :, :) = query
    input(2, :, :) = key_value

    attention = cross_attention_layer(n_heads=1)
    call attention % init([2, 3])
    attention % query_layer % weights = 0.1
    attention % key_layer % weights = 0.1
    attention % value_layer % weights = 0.1
    attention % output_layer % weights = 0.1
    attention % query_layer % biases = 0.11
    attention % key_layer % biases = 0.11
    attention % value_layer % biases = 0.11
    attention % output_layer % biases = 0.11

    call attention % forward(input)
    output_flat = reshape(attention % output, shape(output_flat))
    if (.not. allclose(output_flat, expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if

    call attention % backward(input, gradient)
    query_gradient_flat = reshape(attention % gradient(1, :, :), shape(query_gradient_flat))
    if (.not. allclose(query_gradient_flat, expected_query_gradient_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect query values.. failed'
    end if
    key_value_gradient_flat = reshape(attention % gradient(2, :, :), shape(key_value_gradient_flat))
    if (.not. allclose(key_value_gradient_flat, expected_key_value_gradient_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect key-value values.. failed'
    end if
  end subroutine test_cross_attention
end program test_multihead_attention_layer
