program test_multihead_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_multihead_attention_layer, only: multihead_attention_layer
  use nf_linear2d_layer, only: linear2d_layer
  use nf_optimizers, only: sgd
  implicit none

  logical :: ok = .true.
  type(multihead_attention_layer) :: attention
  real :: sample_input(3, 4) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [3, 4])
  real :: split_heads_output(3, 2, 2)
  real :: minput(3, 4) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [3, 4])
  real :: output(3, 2, 2)

  attention = multihead_attention_layer(sequence_length=3, model_dimension=4, n_heads=2)
  call attention % init([0])
!
  call test_multihead_attention_split_heads(attention, sample_input, ok, split_heads_output)
  call test_multihead_attention_create_attention_matrix(attention, split_heads_output, ok)
  call test_multihead_attention_normalization(attention, ok)
  call test_multihead_attention_scaled_dot_product_attention(attention, split_heads_output, ok)
  call test_multihead_attention_combine_heads(attention, attention % sdpa, ok)
  call test_multihead_attention_forward(attention, ok)
  call test_multihead_attention_backward(attention, ok)
  call test_multihead_attention_update_gradients(attention, ok)
!  call test_multihead_attention_forward_reallife_shape(ok)

contains
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
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: input(:, :, :)
    logical, intent(in out) :: ok
    real :: attention_matrix_shape(3)
    real :: attention_matrix_flat(18)
    real :: expected_shape(3) = [3, 3, 2]
    real :: expected_attention_matrix_flat(18) = [&
        9.00000036E-02, 0.120000005, 0.150000006, 0.120000005, 0.170000002, 0.219999999,&
        0.150000006, 0.219999999, 0.289999992, 1.16999996, 0.518999994, 0.588000000,&
        0.518999994, 0.502099991, 0.573199987, 0.588000000, 0.573199987, 0.654400051&
    ]

    call attention % create_attention_matrix(input, input)

    attention_matrix_shape = shape(attention % attention_matrix)
    if (.not. all(attention_matrix_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'create_attention_matrix returned incorrect shape.. failed'
    end if
    attention_matrix_flat = reshape(attention % attention_matrix, shape(expected_attention_matrix_flat))
    if (.not. all(attention_matrix_flat.eq.expected_attention_matrix_flat)) then
      ok = .false.
      write(stderr, '(a)') 'create_attention_matrix returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_create_attention_matrix

  subroutine test_multihead_attention_normalization(attention, ok)
    type(multihead_attention_layer), intent(in) :: attention
    logical, intent(in out) :: ok
    real :: output_flat(18)
    real :: expected_output_flat(18) = [&
        0.326287806, 0.321620107, 0.316976935, 0.333283335, 0.333194494, 0.333061278,&
        0.340428889, 0.345185429, 0.349961787, 0.435975075, 0.330339372, 0.329200655,&
        0.275134116, 0.326415271, 0.325773478, 0.288890868, 0.343245387, 0.345025837&
    ]

    call attention % normalize_attention_matrix()

    output_flat = reshape(attention % attention_matrix, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'normalize_attention_matrix returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_normalization

  subroutine test_multihead_attention_scaled_dot_product_attention(attention, value, ok)
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: value(:, :, :)
    logical, intent(in out) :: ok
    real :: output_flat(12)
    real :: expected_output_flat(12) = [&
        0.101414114, 0.102356538, 0.103298485, 0.401414126, 0.402356565, 0.403298497,&
        0.685291648, 0.701290667, 0.701582491, 0.457309216, 0.374400556, 0.373518765&
    ]

    call attention % scaled_dot_product_attention(value)

    output_flat = reshape(attention % sdpa, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
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
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'combine_heads returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_combine_heads

  subroutine test_multihead_attention_forward(attention, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    logical, intent(in out) :: ok
    real :: input(3, 4) = reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4])
    real :: output(attention % sequence_length, attention % model_dimension, attention % batch_size)
    real :: output_flat(12)
    integer :: output_shape(2)
    integer :: attn_weights_shape(3)
    real :: attn_weights_flat(18)
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

    call attention % forward(input, input, input)

    output_shape = shape(attention % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
    output_flat = reshape(attention % output, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if

    attn_weights_shape = shape(attention % attention_matrix)
    if (.not. all(attn_weights_shape.eq.expected_attn_weights_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect attention weights shape.. failed'
    end if
    attn_weights_flat = reshape(attention % attention_matrix, shape(attn_weights_flat))
    if (.not. all(attn_weights_flat.eq.expected_attn_weights_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect attention weights values.. failed'
    end if
  end subroutine test_multihead_attention_forward

  subroutine test_multihead_attention_forward_reallife_shape(ok)
    logical, intent(in out) :: ok
    real :: input(148, 512)
    real :: output(148, 512)
    type(linear2d_layer) :: q
    real :: output_flat(12)
    integer :: output_shape(2)
    integer :: expected_shape(2) = [148, 512]
    type(multihead_attention_layer) :: attention

    call random_number(input)

    attention = multihead_attention_layer(sequence_length=148, model_dimension=512, n_heads=8)
    call attention % init([0])

    call attention % forward(input, input, input)

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
    real :: output_flat(12)
    real :: output_shape(2)

    call attention % backward(input, gradient)

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
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_backward

  subroutine test_multihead_attention_update_gradients(attention, ok)
    type(multihead_attention_layer), intent(in out) :: attention
    logical, intent(in out) :: ok
    real :: parameters(80)
    real :: expected_parameters(80)
    real :: updated_output(12)
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

    call attention % forward(&
        reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4]),&
        reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4]),&
        reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4])&
    )

    updated_output = reshape(attention % output, [12])
    if (.not. all(updated_output.eq.expected_updated_output)) then
      ok = .false.
      write(stderr, '(a)') 'incorrect output after parameters update.. failed'
    end if
  end subroutine test_multihead_attention_update_gradients
end program test_multihead_attention_layer