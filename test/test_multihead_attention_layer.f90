program test_multihead_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_multihead_attention_layer, only: multihead_attention_layer
  use nf_linear2d_layer, only: linear2d_layer
  implicit none

  logical :: ok = .true.
  type(multihead_attention_layer) :: attention
  real :: sample_input(3, 4, 1) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [3, 4, 1])
  real :: split_heads_output(2, 3, 2, 1)

  attention = multihead_attention_layer(batch_size=1, sequence_length=3, model_dimension=4, n_heads=2)
  call attention % init([0])

  call test_multihead_attention_split_heads(attention, sample_input, ok, split_heads_output)
  call test_multihead_attention_create_attention_matrix(attention, split_heads_output, ok)
  call test_multihead_attention_normalization(attention, ok)
  call test_multihead_attention_scaled_dot_product_attention(attention, split_heads_output, ok)
  call test_multihead_attention_combine_heads(attention, attention % sdpa, ok)
  call test_multihead_attention_forward(attention, ok)
  call test_multihead_attention_forward_reallife_shape(ok)

contains
  subroutine test_multihead_attention_split_heads(attention, input, ok, output)
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: input(:, :, :)
    logical, intent(in out) :: ok
    real, intent(in out) :: output(2, 3, 2, 1)
    real :: output_shape(4)
    real :: expected_shape(4) = [2, 3, 2, 1]
    real :: output_flat(12)
    real :: expected_output_flat(12) = [0.0, 0.6, 0.1, 0.7, 0.2, 0.8, 0.3, 0.9, 0.4, 0.11, 0.5, 0.12]

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
    real, intent(in) :: input(:, :, :, :)
    logical, intent(in out) :: ok
    real :: attention_matrix_shape(4)
    real :: attention_matrix_flat(18)
    real :: expected_shape(4) = [2, 3, 3, 1]
    real :: expected_attention_matrix_flat(18) = [&
        9.00000036E-02, 1.16999996, 0.120000005,&
        0.518999994, 0.150000006, 0.588000000,&
        0.120000005, 0.518999994, 0.170000002,&
        0.502099991, 0.219999999, 0.573199987,&
        0.150000006, 0.588000000, 0.219999999,&
        0.573199987, 0.289999992, 0.654400051&
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
        0.326287806, 0.435975075, 0.321620107, 0.330339342, 0.316976935, 0.329200655,&
        0.333283335, 0.275134116, 0.333194464, 0.326415271, 0.333061278, 0.325773478,&
        0.340428889, 0.288890868, 0.345185399, 0.343245387, 0.349961787, 0.345025837&
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
    real, intent(in) :: value(:, :, :, :)
    logical, intent(in out) :: ok
    real :: output_flat(12)
    real :: expected_output_flat(12) = [&
        0.101414114, 0.685291648, 0.102356531, 0.701290607, 0.103298485, 0.701582491,&
        0.401414126, 0.457309216, 0.402356505, 0.374400526, 0.403298497, 0.373518765&
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
    real, intent(in) :: scaled_dp_att(:, :, :, :)
    logical, intent(in out) :: ok
    real :: output(attention % sequence_length, attention % model_dimension, attention % batch_size)
    real :: output_flat(12)
    real :: expected_output_flat(12) = [&
        0.101414114, 0.102356531, 0.103298485, 0.401414126, 0.402356505, 0.403298497,&
        0.685291648, 0.701290607, 0.701582491, 0.457309216, 0.374400526, 0.373518765&
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
    real :: input(3, 4, 1) = reshape([0.0, 10.1, 0.2, 10.3, 0.4, 10.5, 0.6, 10.7, 10.8, 0.9, 0.11, 0.12], [3, 4, 1])
    real :: output(attention % sequence_length, attention % model_dimension, attention % batch_size)
    real :: output_flat(12)
    integer :: output_shape(3)
    integer :: expected_shape(3) = [3, 4, 1]
    real :: expected_output_flat(12) = [&
        0.982241452, 1.00407875, 1.00444126, 0.982241452, 1.00407875, 1.00444126,&
        0.982241452, 1.00407875, 1.00444126, 0.982241452, 1.00407875, 1.00444126&
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
  end subroutine test_multihead_attention_forward

  subroutine test_multihead_attention_forward_reallife_shape(ok)
    logical, intent(in out) :: ok
    real :: input(148, 512, 2)
    real :: output(148, 512, 2)
    type(linear2d_layer) :: q
    real :: output_flat(12)
    integer :: output_shape(3)
    integer :: expected_shape(3) = [148, 512, 2]
    type(multihead_attention_layer) :: attention

    call random_number(input)

    attention = multihead_attention_layer(batch_size=2, sequence_length=148, model_dimension=512, n_heads=8)
    call attention % init([0])

    call attention % forward(input, input, input)

    output_shape = shape(attention % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
  end subroutine test_multihead_attention_forward_reallife_shape
end program test_multihead_attention_layer